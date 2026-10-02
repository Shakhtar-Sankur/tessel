// Runs, on this Mac's GPU, the kernels examples/metal_cases.rs writes:
// for each case, compiles DIR/CASE/kernel.metal with Metal, binds
// DIR/CASE/in<i>.bin as buffer i, launches it and writes every buffer back
// as DIR/CASE/out<i>.bin, for `metal_cases check DIR` to compare with the
// reference interpreter.
//
// usage: swift scripts/metal_run.swift DIR
// Exits with 3, having run nothing, where there is no Metal device (some
// virtual machines).

import Foundation
import Metal

struct Case: Decodable {
    let name: String
    let function: String
    let args: Int
    let grid: [Int]
    let threads: Int
    enum CodingKeys: String, CodingKey {
        case name = "case", function, args, grid, threads
    }
}

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(1)
}

let argv = CommandLine.arguments
guard argv.count == 2 else { fail("usage: swift scripts/metal_run.swift DIR") }
let dir = URL(fileURLWithPath: argv[1])
let cases: [Case]
do {
    cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: dir.appendingPathComponent("cases.json")))
} catch {
    fail("reading cases.json: \(error)")
}
guard let dev = MTLCreateSystemDefaultDevice() else {
    print("no Metal device")
    exit(3)
}
print("device: \(dev.name)")
guard let queue = dev.makeCommandQueue() else { fail("no command queue") }

for c in cases {
    let cd = dir.appendingPathComponent(c.name)
    do {
        let src = try String(contentsOf: cd.appendingPathComponent("kernel.metal"), encoding: .utf8)
        let opts = MTLCompileOptions()
        // The kernels compare against infinity, as IEEE arithmetic defines it.
        opts.fastMathEnabled = false
        let lib = try dev.makeLibrary(source: src, options: opts)
        guard let fn = lib.makeFunction(name: c.function) else { fail("\(c.name): no function \(c.function)") }
        let pso = try dev.makeComputePipelineState(function: fn)
        // The kernels are written for SIMD-groups of 32 threads, as warps.
        guard pso.threadExecutionWidth == 32 else {
            fail("\(c.name): SIMD-groups of \(pso.threadExecutionWidth) threads; the kernels need 32")
        }
        guard pso.maxTotalThreadsPerThreadgroup >= c.threads else {
            fail("\(c.name): at most \(pso.maxTotalThreadsPerThreadgroup) threads per threadgroup; the kernel needs \(c.threads)")
        }
        var bufs: [MTLBuffer] = []
        for i in 0..<c.args {
            let d = try Data(contentsOf: cd.appendingPathComponent("in\(i).bin"))
            let b = d.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: d.count, options: .storageModeShared) }
            guard let b else { fail("\(c.name): allocating buffer \(i)") }
            bufs.append(b)
        }
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            fail("\(c.name): no command buffer")
        }
        enc.setComputePipelineState(pso)
        for (i, b) in bufs.enumerated() {
            enc.setBuffer(b, offset: 0, index: i)
        }
        enc.dispatchThreadgroups(
            MTLSize(width: c.grid[0], height: c.grid[1], depth: c.grid[2]),
            threadsPerThreadgroup: MTLSize(width: c.threads, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { fail("\(c.name): \(e)") }
        for (i, b) in bufs.enumerated() {
            try Data(bytes: b.contents(), count: b.length).write(to: cd.appendingPathComponent("out\(i).bin"))
        }
        print("\(c.name): ran \(c.grid) threadgroups of \(c.threads) threads")
    } catch {
        fail("\(c.name): \(error)")
    }
}
