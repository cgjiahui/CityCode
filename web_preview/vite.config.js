import { defineConfig } from 'vite';
export default defineConfig({base:'./',server:{proxy:{'/api':'http://127.0.0.1:8794','/data':'http://127.0.0.1:8794'}}});
