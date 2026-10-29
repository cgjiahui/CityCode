#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
struct Rays {std::vector<float> lo,hi,hits;std::vector<int64_t> offsets;};
extern "C" {
void* raster_create(const double* v,const int32_t* f,int64_t nf,double ox,double oy,double cell,int w,int h){
 auto* r=new Rays;size_t n=size_t(w)*h;std::vector<std::vector<float>> bins(n);
 for(int64_t ti=0;ti<nf;ti++){
  const double *a=v+3*f[3*ti],*b=v+3*f[3*ti+1],*c=v+3*f[3*ti+2];
  double bx=b[0]-a[0],by=b[1]-a[1],cx=c[0]-a[0],cy=c[1]-a[1],den=bx*cy-by*cx;
  if(std::abs(den)<1e-14)continue;
  int x0=std::max(0,int(std::ceil((std::min({a[0],b[0],c[0]})-ox)/cell-.5-1e-8)));
  int x1=std::min(w-1,int(std::floor((std::max({a[0],b[0],c[0]})-ox)/cell-.5+1e-8)));
  int y0=std::max(0,int(std::ceil((std::min({a[1],b[1],c[1]})-oy)/cell-.5-1e-8)));
  int y1=std::min(h-1,int(std::floor((std::max({a[1],b[1],c[1]})-oy)/cell-.5+1e-8)));
  for(int y=y0;y<=y1;y++)for(int x=x0;x<=x1;x++){
   double px=ox+(x+.5)*cell-a[0],py=oy+(y+.5)*cell-a[1];
   double u=(px*cy-py*cx)/den,t=(bx*py-by*px)/den;
   if(u>=-1e-9 && t>=-1e-9 && u+t<=1+1e-9)bins[size_t(y)*w+x].push_back(float(a[2]+u*(b[2]-a[2])+t*(c[2]-a[2])));
  }
 }
 r->lo.resize(n,std::numeric_limits<float>::quiet_NaN());r->hi=r->lo;r->offsets.resize(n+1);r->offsets[0]=0;
 for(size_t i=0;i<n;i++){
  auto& z=bins[i];std::sort(z.begin(),z.end());float last=-std::numeric_limits<float>::infinity();
  for(float v:z)if(v-last>.002f){r->hits.push_back(v);last=v;}
  if(r->hits.size()>size_t(r->offsets[i])){r->lo[i]=r->hits[r->offsets[i]];r->hi[i]=r->hits.back();}
  r->offsets[i+1]=r->hits.size();
 }
 return r;
}
int64_t raster_hit_count(void* p){return ((Rays*)p)->hits.size();}
const float* raster_hits(void* p){return ((Rays*)p)->hits.data();}
const float* raster_lo(void* p){return ((Rays*)p)->lo.data();}
const float* raster_hi(void* p){return ((Rays*)p)->hi.data();}
const int64_t* raster_offsets(void* p){return ((Rays*)p)->offsets.data();}
void raster_destroy(void* p){delete (Rays*)p;}
}
