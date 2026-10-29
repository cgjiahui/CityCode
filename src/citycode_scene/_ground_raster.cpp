#include <algorithm>
#include <cmath>
#include <cstdint>

// Vertices are already transformed into world XYZ metres, with Z up. The
// raster contains maximum supporting height per semantic role, not all layers.
extern "C" void ground_rasterize(
    const double* vertices, const int32_t* faces, int64_t nf,
    int64_t width, int64_t height, double x0, double y0, double cell,
    int mode, double minimum_normal_z, double near_limit, double fallback_cap,
    float* continuous, float* detail_near, float* detail_max) {
    const double invcell = 1.0 / cell;
    for (int64_t fi = 0; fi < nf; ++fi) {
        const double* a = vertices + int64_t(faces[3*fi])*3;
        const double* b = vertices + int64_t(faces[3*fi+1])*3;
        const double* c = vertices + int64_t(faces[3*fi+2])*3;
        const double ux=b[0]-a[0], uy=b[1]-a[1], uz=b[2]-a[2];
        const double vx=c[0]-a[0], vy=c[1]-a[1], vz=c[2]-a[2];
        const double det=ux*vy-uy*vx;
        if (std::abs(det)<1e-12) continue;
        const double nx=uy*vz-uz*vy, ny=uz*vx-ux*vz;
        if (det*det < minimum_normal_z*minimum_normal_z*(nx*nx+ny*ny+det*det)) continue;
        const int64_t xmin=std::max<int64_t>(0, std::ceil((std::min({a[0],b[0],c[0]})-x0)*invcell-.5-1e-8));
        const int64_t xmax=std::min<int64_t>(width-1, std::floor((std::max({a[0],b[0],c[0]})-x0)*invcell-.5+1e-8));
        const int64_t ymin=std::max<int64_t>(0, std::ceil((std::min({a[1],b[1],c[1]})-y0)*invcell-.5-1e-8));
        const int64_t ymax=std::min<int64_t>(height-1, std::floor((std::max({a[1],b[1],c[1]})-y0)*invcell-.5+1e-8));
        for (int64_t iy=ymin; iy<=ymax; ++iy) {
            const double py=y0+(iy+.5)*cell-a[1];
            for (int64_t ix=xmin; ix<=xmax; ++ix) {
                const double px=x0+(ix+.5)*cell-a[0];
                const double u=(px*vy-py*vx)/det, v=(ux*py-uy*px)/det;
                if (u< -1e-7 || v< -1e-7 || u+v>1+1e-7) continue;
                const float z=float(a[2]+u*uz+v*vz);
                const int64_t j=iy*width+ix;
                if (mode==0) {
                    if (z>continuous[j]) continuous[j]=z;
                } else {
                    if (z>detail_max[j]) detail_max[j]=z;
                    const double ref=std::isfinite(continuous[j]) ? continuous[j] : fallback_cap;
                    if (std::isfinite(ref) && z<=ref+near_limit && z>detail_near[j]) detail_near[j]=z;
                }
            }
        }
    }
}
