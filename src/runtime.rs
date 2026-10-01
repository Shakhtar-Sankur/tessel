//! Running compiled kernels: on kiln's emulator of the CUDA execution model
//! (the generated source compiled as C++ for the CPU), or on an NVIDIA GPU
//! through NVRTC and the driver API.

use crate::ast::DType;
use crate::cuda::{self, Generated, Options};
use crate::driver::{Cuda, DevPtr, Function, Nvrtc};
use crate::half::{f16_to_f32, f32_to_f16};
use crate::interp::{Tensor, check_args};
use crate::ir::{Kernel, PKind};
use crate::jit;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Emu,
    Cuda,
}

type EmuFn = unsafe extern "C" fn(*mut *mut f32, u32, u32, u32, u32, u32);

enum Exec {
    Emu(EmuFn, #[allow(dead_code)] jit::Library),
    Cuda(Function),
}

pub struct Compiled {
    pub kernel: Kernel,
    pub code: Generated,
    exec: Exec,
    pub compile_seconds: f64,
}

/// Flags for the emulator build.
const EMU_FLAGS: &[&str] = &["-O1", "-std=c++17", "-fPIC", "-shared", "-pthread", "-w"];

static CUDA: OnceLock<Result<Cuda, String>> = OnceLock::new();
static NVRTC: OnceLock<Result<Nvrtc, String>> = OnceLock::new();

/// The GPU, opened once per process.
pub fn cuda() -> Result<&'static Cuda, String> {
    let c = CUDA.get_or_init(Cuda::open).as_ref().map_err(|e| e.clone())?;
    c.bind()?;
    Ok(c)
}

pub fn nvrtc() -> Result<&'static Nvrtc, String> {
    NVRTC.get_or_init(Nvrtc::open).as_ref().map_err(|e| e.clone())
}

/// CUBIN for `src` on sm_{arch}, cached on disk by content.
pub fn cubin(src: &str, arch: u32) -> Result<Vec<u8>, String> {
    let nv = nvrtc()?;
    let key = format!("{:016x}", jit::fnv(&format!("sm_{arch} {:?}\n{src}", nv.version)));
    let dir = jit::cache_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!("g{key}.cubin"));
    if let Ok(b) = std::fs::read(&path) {
        return Ok(b);
    }
    let (bin, _) = nv.compile(src, arch, false)?;
    let tmp = dir.join(format!("g{key}.{}.tmp", std::process::id()));
    std::fs::write(&tmp, &bin).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
    Ok(bin)
}

/// Compiles `k` for `dev`.
pub fn compile(k: &Kernel, dev: Device, o: &Options) -> Result<Compiled, String> {
    let t = std::time::Instant::now();
    let mut o = o.clone();
    if dev == Device::Cuda {
        let c = cuda()?;
        o.arch = (c.cc.0 * 10 + c.cc.1) as u32;
    }
    let code = cuda::generate(k, &o)?;
    let exec = match dev {
        Device::Emu => {
            let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
            let lib = jit::compile_with(&cxx, EMU_FLAGS, &["-lm"], "cpp", &code.source)?;
            let p = lib.symbol(&format!("{}_emu", code.name))?;
            // SAFETY: every emulator entry point has this signature.
            let f = unsafe { std::mem::transmute::<*mut std::ffi::c_void, EmuFn>(p) };
            Exec::Emu(f, lib)
        }
        Device::Cuda => {
            let c = cuda()?;
            let bin = cubin(&code.source, o.arch)?;
            let f = c.load(&bin, std::slice::from_ref(&code.name))?[0];
            if code.smem > 48 * 1024 {
                c.allow_smem(f, code.smem as u32).map_err(|_| {
                    format!(
                        "needs {} KB of shared memory, more than this GPU allows a block",
                        code.smem.div_ceil(1024)
                    )
                })?;
            }
            Exec::Cuda(f)
        }
    };
    Ok(Compiled {
        kernel: k.clone(),
        code,
        exec,
        compile_seconds: t.elapsed().as_secs_f64(),
    })
}

/// A tensor's bytes as the kernel reads them.
pub fn to_bytes(t: &Tensor) -> Vec<u8> {
    let mut b = Vec::with_capacity(t.data.len() * t.dtype.bytes());
    for &x in &t.data {
        match t.dtype {
            DType::F32 => b.extend_from_slice(&x.to_le_bytes()),
            DType::F16 => b.extend_from_slice(&f32_to_f16(x).to_le_bytes()),
            DType::I32 => b.extend_from_slice(&(x as i32).to_le_bytes()),
            DType::Bool => b.push((x != 0.0) as u8),
        }
    }
    b
}

pub fn from_bytes(t: &mut Tensor, b: &[u8]) {
    let w = t.dtype.bytes();
    for (i, x) in t.data.iter_mut().enumerate() {
        let c = &b[i * w..(i + 1) * w];
        *x = match t.dtype {
            DType::F32 => f32::from_le_bytes(c.try_into().unwrap()),
            DType::F16 => f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())),
            DType::I32 => i32::from_le_bytes(c.try_into().unwrap()) as f32,
            DType::Bool => c[0] as f32,
        };
    }
}

/// Device copies of a kernel's arguments, for repeated launches.
pub struct DeviceArgs {
    ptrs: Vec<DevPtr>,
    bytes: Vec<Vec<u8>>,
}

impl Compiled {
    /// Runs the kernel once; arrays it writes are copied back into `args`.
    pub fn run(&self, args: &mut [Tensor]) -> Result<(), String> {
        check_args(&self.kernel, args)?;
        let g = &self.code;
        match &self.exec {
            Exec::Emu(f, _) => {
                // 16-byte aligned copies, so vector accesses behave as on the GPU.
                let mut bufs: Vec<Vec<u128>> = args
                    .iter()
                    .map(|t| {
                        let b = to_bytes(t);
                        let mut v = vec![0u128; b.len().div_ceil(16).max(1)];
                        // SAFETY: v has at least b.len() bytes.
                        unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), v.as_mut_ptr() as *mut u8, b.len()) };
                        v
                    })
                    .collect();
                let mut ptrs: Vec<*mut f32> = bufs.iter_mut().map(|b| b.as_mut_ptr() as *mut f32).collect();
                // SAFETY: the pointers match the kernel's parameters and live
                // for the call.
                unsafe {
                    f(
                        ptrs.as_mut_ptr(),
                        g.grid[0] as u32,
                        g.grid[1] as u32,
                        g.grid[2] as u32,
                        g.threads as u32,
                        g.smem as u32,
                    )
                };
                for (t, b) in args.iter_mut().zip(&bufs) {
                    if !t.shape.is_empty() || matches!(self.kernel.params.len(), 0) {
                        let n = t.data.len() * t.dtype.bytes();
                        // SAFETY: b holds at least n bytes.
                        let bytes = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, n) };
                        from_bytes(t, bytes);
                    }
                }
                Ok(())
            }
            Exec::Cuda(_) => {
                let d = self.upload(args)?;
                self.launch(&d)?;
                self.download(&d, args)
            }
        }
    }

    pub fn upload(&self, args: &[Tensor]) -> Result<DeviceArgs, String> {
        let c = cuda()?;
        let mut ptrs = Vec::new();
        let mut bytes = Vec::new();
        for (p, t) in self.kernel.params.iter().zip(args) {
            let b = to_bytes(t);
            match p.kind {
                PKind::Array { .. } => {
                    let d = c.alloc(b.len())?;
                    c.upload_bytes(d, &b)?;
                    ptrs.push(d);
                }
                PKind::Scalar(_) => {
                    let mut v = [0u8; 8];
                    v[..b.len()].copy_from_slice(&b);
                    ptrs.push(u64::from_le_bytes(v));
                }
            }
            bytes.push(b);
        }
        Ok(DeviceArgs { ptrs, bytes })
    }

    pub fn launch(&self, d: &DeviceArgs) -> Result<(), String> {
        let Exec::Cuda(f) = &self.exec else {
            return Err("not a GPU kernel".into());
        };
        let g = &self.code;
        cuda()?.launch(
            *f,
            [g.grid[0] as u32, g.grid[1] as u32, g.grid[2] as u32],
            g.threads as u32,
            g.smem as u32,
            &d.ptrs,
        )
    }

    pub fn download(&self, d: &DeviceArgs, args: &mut [Tensor]) -> Result<(), String> {
        let c = cuda()?;
        c.sync()?;
        for ((p, t), (ptr, b)) in self
            .kernel
            .params
            .iter()
            .zip(args.iter_mut())
            .zip(d.ptrs.iter().zip(&d.bytes))
        {
            if let PKind::Array { .. } = p.kind {
                let mut out = vec![0u8; b.len()];
                c.download_bytes(&mut out, *ptr)?;
                from_bytes(t, &out);
            }
        }
        Ok(())
    }

    /// Median and minimum milliseconds of `iters` launches (after warmup).
    pub fn time(&self, d: &DeviceArgs, iters: usize) -> Result<(f64, f64), String> {
        let c = cuda()?;
        // Warm up for at least 50 ms of launches (and 3), so the clocks have
        // ramped up after any idle time (compiling, say); the baselines'
        // timing does the same.
        let t = std::time::Instant::now();
        let mut n = 0;
        while n < 3 || t.elapsed().as_millis() < 50 {
            self.launch(d)?;
            n += 1;
            if n % 8 == 0 {
                c.sync()?;
            }
        }
        c.sync()?;
        let mut ts = Vec::new();
        for _ in 0..iters {
            ts.push(c.time(&mut || self.launch(d))? as f64);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok((ts[ts.len() / 2], ts[0]))
    }
}
