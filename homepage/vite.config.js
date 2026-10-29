import { defineConfig } from 'vite';
import {copyFileSync,mkdirSync,readFileSync} from 'node:fs';

const basisFiles=['basis_transcoder.js','basis_transcoder.wasm','README.md'];
let outputDir='dist';

export default defineConfig({
  base: './',
  build: {outDir: '.frontend-build'},
  plugins:[{name:'local-basis-decoder',
    configResolved(config){outputDir=config.build.outDir;},
    configureServer(server){server.middlewares.use('/basis',(req,res,next)=>{const name=(req.url||'').split('?')[0].replace(/^\//,'');if(!basisFiles.includes(name))return next();res.setHeader('Content-Type',name.endsWith('.wasm')?'application/wasm':name.endsWith('.js')?'application/javascript':'text/plain');res.end(readFileSync('node_modules/three/examples/jsm/libs/basis/'+name));});},
    closeBundle(){mkdirSync(outputDir+'/basis',{recursive:true});for(const name of basisFiles)copyFileSync('node_modules/three/examples/jsm/libs/basis/'+name,outputDir+'/basis/'+name);}}],
  server: { proxy: { '/city/': 'http://127.0.0.1:8798' } },
});
