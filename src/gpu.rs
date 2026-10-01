//! Buffers that outlive a kernel launch, and launches that take them: what
//! a program made of many kernels (the LLM engine) runs on. The same code
//! drives the emulator, where buffers are host memory, and an NVIDIA GPU.

use crate::cuda::Options;
use crate::interp::Tensor;
use crate::ir::{self, Spec};
use crate::kernels;
use crate::runtime::{self, Compiled, Device};
use std::collections::HashMap;

/// A buffer: its index and size in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buf {
    id: usize,
    pub bytes: usize,
}

/// One kernel argument.
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    Buf(Buf),
    F32(f32),
    I32(i32),
}

pub struct Gpu {
    pub dev: Device,
    /// Emulator buffers (16-byte aligned), or GPU allocations.
    host: Vec<Vec<u128>>,
    dptr: Vec<u64>,
    kernels: Vec<Compiled>,
    by_key: HashMap<String, usize>,
    /// Scalars passed to the emulator live here during a launch.
    scalars: Vec<u128>,
    pub launches: u64,
}

/// A compiled kernel of a `Gpu`.
#[derive(Clone, Copy, Debug)]
pub struct K(usize);

impl Gpu {
    pub fn new(dev: Device) -> Result<Gpu, String> {
        if dev == Device::Cuda {
            runtime::cuda()?;
        }
        Ok(Gpu {
            dev,
            host: Vec::new(),
            dptr: Vec::new(),
            kernels: Vec::new(),
            by_key: HashMap::new(),
            scalars: vec![0; 64],
            launches: 0,
        })
    }

    /// A zeroed buffer of `bytes` bytes.
    pub fn alloc(&mut self, bytes: usize) -> Result<Buf, String> {
        let id = match self.dev {
            Device::Emu => {
                self.host.push(vec![0u128; bytes.div_ceil(16).max(1)]);
                self.host.len() - 1
            }
            Device::Cuda => {
                let c = runtime::cuda()?;
                let p = c.alloc(bytes)?;
                c.upload_bytes(p, &vec![0u8; bytes])?;
                self.dptr.push(p);
                self.dptr.len() - 1
            }
        };
        Ok(Buf { id, bytes })
    }

    /// Copies `data` into `b` at byte `offset`.
    pub fn write(&mut self, b: Buf, offset: usize, data: &[u8]) -> Result<(), String> {
        if offset + data.len() > b.bytes {
            return Err(format!(
                "write of {} bytes at {offset} into a buffer of {}",
                data.len(),
                b.bytes
            ));
        }
        match self.dev {
            Device::Emu => {
                let h = &mut self.host[b.id];
                // SAFETY: the range lies inside the buffer (checked above).
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), (h.as_mut_ptr() as *mut u8).add(offset), data.len())
                };
                Ok(())
            }
            Device::Cuda => runtime::cuda()?.upload_bytes(self.dptr[b.id] + offset as u64, data),
        }
    }

    /// Reads `out.len()` bytes of `b` from byte `offset` (waiting for the GPU).
    pub fn read(&self, b: Buf, offset: usize, out: &mut [u8]) -> Result<(), String> {
        if offset + out.len() > b.bytes {
            return Err(format!(
                "read of {} bytes at {offset} from a buffer of {}",
                out.len(),
                b.bytes
            ));
        }
        match self.dev {
            Device::Emu => {
                let h = &self.host[b.id];
                // SAFETY: the range lies inside the buffer (checked above).
                unsafe {
                    std::ptr::copy_nonoverlapping((h.as_ptr() as *const u8).add(offset), out.as_mut_ptr(), out.len())
                };
                Ok(())
            }
            Device::Cuda => {
                let c = runtime::cuda()?;
                c.sync()?;
                c.download_bytes(out, self.dptr[b.id] + offset as u64)
            }
        }
    }

    pub fn read_f32(&self, b: Buf, offset_elems: usize, n: usize) -> Result<Vec<f32>, String> {
        let mut bytes = vec![0u8; n * 4];
        self.read(b, offset_elems * 4, &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// Kernel `name` of built-in file `file`, compiled for these shapes and
    /// settings once and then reused.
    pub fn kernel(
        &mut self,
        file: &str,
        name: &str,
        shapes: &[Vec<usize>],
        meta: &[(&str, i64)],
        warps: usize,
    ) -> Result<K, String> {
        let key = format!("{file}/{name} {shapes:?} {meta:?} {warps}");
        if let Some(&k) = self.by_key.get(&key) {
            return Ok(K(k));
        }
        let src = kernels::source(file).ok_or_else(|| format!("no kernel file {file}"))?;
        let spec = Spec {
            shapes: shapes.to_vec(),
            meta: meta.iter().map(|(n, v)| (n.to_string(), *v)).collect(),
        };
        let k = ir::compile(src, name, &spec)?;
        let c = runtime::compile(&k, self.dev, &Options { warps, arch: 75 })?;
        self.kernels.push(c);
        self.by_key.insert(key, self.kernels.len() - 1);
        Ok(K(self.kernels.len() - 1))
    }

    pub fn compiled(&self, k: K) -> &Compiled {
        &self.kernels[k.0]
    }

    /// Launches `k` (asynchronously on the GPU).
    pub fn launch(&mut self, k: K, args: &[Arg]) -> Result<(), String> {
        let mut raw = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            raw.push(match (self.dev, a) {
                (Device::Emu, Arg::Buf(b)) => self.host[b.id].as_mut_ptr() as u64,
                (Device::Cuda, Arg::Buf(b)) => self.dptr[b.id],
                (Device::Emu, Arg::F32(x)) => {
                    self.scalars[i] = x.to_bits() as u128;
                    &self.scalars[i] as *const u128 as u64
                }
                (Device::Emu, Arg::I32(x)) => {
                    self.scalars[i] = *x as u32 as u128;
                    &self.scalars[i] as *const u128 as u64
                }
                (Device::Cuda, Arg::F32(x)) => x.to_bits() as u64,
                (Device::Cuda, Arg::I32(x)) => *x as u32 as u64,
            });
        }
        self.launches += 1;
        self.kernels[k.0].launch_raw(&raw)
    }

    pub fn sync(&self) -> Result<(), String> {
        match self.dev {
            Device::Emu => Ok(()),
            Device::Cuda => runtime::cuda()?.sync(),
        }
    }

    /// A buffer holding `t` as the kernels read it.
    pub fn tensor(&mut self, t: &Tensor) -> Result<Buf, String> {
        let b = runtime::to_bytes(t);
        let buf = self.alloc(b.len())?;
        self.write(buf, 0, &b)?;
        Ok(buf)
    }
}
