import * as T from 'three';

// A static sky costs one draw call and no image download or continuous animation.
// Z is up in this scene. The view rotation makes the sky follow street-level looks.
export function addAtmosphere(scene) {
  const sunDirection=new T.Vector3(-600,-900,1600).normalize();
  const sky=new T.Mesh(new T.SphereGeometry(1,32,16),new T.ShaderMaterial({
    side:T.BackSide,depthWrite:false,depthTest:false,fog:false,toneMapped:false,
    uniforms:{
      zenith:{value:new T.Color('#82acd0')},horizon:{value:new T.Color('#e5edf2')},
      cloudColor:{value:new T.Color('#fffaf1')},sunDirection:{value:sunDirection},
    },
    vertexShader:`varying vec3 skyDirection;
      void main(){skyDirection=position;
        vec4 p=projectionMatrix*vec4(mat3(viewMatrix)*position,1.0);
        gl_Position=p.xyww;}`,
    fragmentShader:`
      varying vec3 skyDirection;
      uniform vec3 zenith,horizon,cloudColor,sunDirection;
      float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
      float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.0-2.0*f);
        return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);}
      float cloud(vec2 p){return noise(p)*.57+noise(p*2.03+7.4)*.28+noise(p*4.11)*.15;}
      void main(){vec3 d=normalize(skyDirection);float elevation=max(d.z,0.0);
        vec3 color=mix(horizon,zenith,pow(elevation,.52));
        float sun=max(dot(d,sunDirection),0.0);
        color=mix(color,cloudColor,pow(sun,18.0)*.14);
        vec2 uv=d.xy/max(.13,d.z)*vec2(1.25,2.1)+vec2(13.2,4.7);
        float coverage=smoothstep(.50,.78,cloud(uv));
        coverage*=smoothstep(.035,.24,d.z)*(1.0-smoothstep(.8,1.0,d.z));
        color=mix(color,cloudColor,coverage*.64);
        gl_FragColor=vec4(color,1.0);
        #include <colorspace_fragment>
      }`,
  }));
  sky.name='Daylight sky';sky.frustumCulled=false;sky.renderOrder=-1000;
  scene.background=null;scene.fog=new T.Fog('#e5edf2',3200,6500);scene.add(sky);
  const ambient=new T.HemisphereLight('#edf4ff','#b6b0a5',2.2);ambient.position.set(0,0,1);
  const sun=new T.DirectionalLight('#fff2de',2.0);sun.position.copy(sunDirection).multiplyScalar(1600);
  const fill=new T.DirectionalLight('#dbe9f7',.8);fill.position.set(700,800,500);
  // Diffuse daylight preserves the captured facade colors; no reflective HDR sky
  // or full-city shadow render pass is added to the already large scene.
  scene.add(ambient,sun,fill);
}

// Low-frequency sky reflection for native PBR ground/props only. Building
// materials keep their accepted illumination and never receive this env map.
export function createMaterialEnvironment(renderer){
  const canvas=document.createElement('canvas');canvas.width=512;canvas.height=256;
  const context=canvas.getContext('2d'),gradient=context.createLinearGradient(0,0,0,256);
  gradient.addColorStop(0,'#9db7cd');gradient.addColorStop(.45,'#e9eff2');gradient.addColorStop(.52,'#c8c8c1');gradient.addColorStop(1,'#646963');
  context.fillStyle=gradient;context.fillRect(0,0,512,256);
  const source=new T.CanvasTexture(canvas);source.mapping=T.EquirectangularReflectionMapping;source.colorSpace=T.SRGBColorSpace;
  const generator=new T.PMREMGenerator(renderer),target=generator.fromEquirectangular(source);
  generator.dispose();source.dispose();return target;
}
