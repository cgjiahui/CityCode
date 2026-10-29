#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
// Diagnostic only: all active triangles, both windings, nearest orthographic surface.
extern "C" void surface(const double* v,const int64_t* f,int64_t nf,double ox,double oy,double cell,int w,int h,float* z,int64_t* ids,float* bu,float* bv){
 const int64_t n=int64_t(w)*h;
 std::fill(z,z+n,-std::numeric_limits<float>::infinity());std::fill(ids,ids+n,-1);
 for(int64_t k=0;k<nf;++k){
  const double *a=v+3*f[3*k],*b=v+3*f[3*k+1],*c=v+3*f[3*k+2];
  double d=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);if(std::abs(d)<1e-14)continue;
  int x0=std::max(0,int(std::ceil((std::min({a[0],b[0],c[0]})-ox)/cell-.5))),x1=std::min(w-1,int(std::floor((std::max({a[0],b[0],c[0]})-ox)/cell-.5)));
  int y0=std::max(0,int(std::ceil((std::min({a[1],b[1],c[1]})-oy)/cell-.5))),y1=std::min(h-1,int(std::floor((std::max({a[1],b[1],c[1]})-oy)/cell-.5)));
  for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x){double px=ox+(x+.5)*cell,py=oy+(y+.5)*cell;
   double u=((b[1]-c[1])*(px-c[0])+(c[0]-b[0])*(py-c[1]))/d,q=((c[1]-a[1])*(px-c[0])+(a[0]-c[0])*(py-c[1]))/d;
   if(u>=-1e-9&&q>=-1e-9&&u+q<=1+1e-9){float zz=float(u*a[2]+q*b[2]+(1-u-q)*c[2]);int64_t i=int64_t(y)*w+x;
    if(zz>z[i]){z[i]=zz;ids[i]=k;bu[i]=float(u);bv[i]=float(q);}}
  }
 }
}
