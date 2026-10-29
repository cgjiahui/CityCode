// MIT meshoptimizer 0.25 CPU/WASM; one finite invocation, indices only.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const dependency = path.join(__dirname, 'vendor', 'meshoptimizer_v025');
const snapshot = JSON.parse(fs.readFileSync(path.join(dependency, 'SNAPSHOT.json')));
for (const record of snapshot.files) {
  const data = fs.readFileSync(path.join(dependency, record.path));
  if (data.length !== record.bytes || crypto.createHash('sha256').update(data).digest('hex') !== record.sha256)
    throw new Error('Meshoptimizer dependency differs from its packaged snapshot');
}
const MeshoptSimplifier = require(path.join(dependency, 'meshopt_simplifier.js'));
const directory = process.argv[2];
if (!directory) throw new Error('A private input/output directory is required');
function load(name, Type) {
  const data = fs.readFileSync(path.join(directory, name));
  return new Type(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
}
(async () => {
  await MeshoptSimplifier.ready;
  if (!MeshoptSimplifier.supported) throw new Error('CPU WebAssembly is unavailable');
  const config = JSON.parse(fs.readFileSync(path.join(directory, 'config.json')));
  const indices = load('indices.u32', Uint32Array);
  const positions = load('positions.f32', Float32Array);
  const attributes = load('attributes.f32', Float32Array);
  const locks = load('locks.u8', Uint8Array);
  const started = performance.now();
  const [result, error] = MeshoptSimplifier.simplifyWithAttributes(
    indices, positions, 3, attributes, 5, [1, 1, 1, 1, 1], locks,
    config.target_faces * 3, config.target_relative_error ?? 0.01, []);
  fs.writeFileSync(path.join(directory, 'simplified_indices.u32'), Buffer.from(result.buffer, result.byteOffset, result.byteLength));
  fs.writeFileSync(path.join(directory, 'backend_result.json'), JSON.stringify({
    backend: 'meshoptimizer', version: snapshot.version, method: 'simplifyWithAttributes',
    node_version: process.versions.node, target_faces: config.target_faces,
    target_relative_error: config.target_relative_error ?? 0.01, reported_error: error,
    attribute_weights: [1, 1, 1, 1, 1], flags: [], invocation_count: 1,
    input_faces: indices.length / 3, output_faces: result.length / 3,
    vertex_attribute_updates: false,
    elapsed_seconds: (performance.now() - started) / 1000
  }, null, 2) + '\n');
})().catch(error => { console.error(error.message); process.exitCode = 1; });
