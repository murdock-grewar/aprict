struct config_CB_project {
};

struct uivec3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
};
__device__ uivec3 operator+(
        uivec3 v1,
        uivec3 v2
)
{
    uivec3 v3;
    v3.x = v1.x + v2.x;
    v3.y = v1.y + v2.y;
    v3.z = v1.z + v2.z;
    return v3;
}


__device__ uivec3 operator-(
        uivec3 v1,
        uivec3 v2
)
{
    uivec3 v3;
    v3.x = v1.x - v2.x;
    v3.y = v1.y - v2.y;
    v3.z = v1.z - v2.z;
    return v3;
}


__device__ float dot_prod(
        uivec3 v1,
        uivec3 v2
)
{
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}



struct fvec3 {
    float x;
    float y;
    float z;

public:
    // implicit constructor from floats
    fvec3(float xi, float yi, float zi) : x(xi), y(yi), z(zi) {}
    // implicit constructor from a uivec3
    fvec3(uivec3 v) : x(v.x), y(v.y), z(v.z) {}
    // default constructor, used with i.e. 'fvec3 v;'
    fvec3() = default;
};
__device__ fvec3 operator+(
        fvec3 v1,
        fvec3 v2
)
{
    fvec3 v3;
    v3.x = v1.x + v2.x;
    v3.y = v1.y + v2.y;
    v3.z = v1.z + v2.z;
    return v3;
}


__device__ fvec3 operator-(
        fvec3 v1,
        fvec3 v2
)
{
    fvec3 v3;
    v3.x = v1.x - v2.x;
    v3.y = v1.y - v2.y;
    v3.z = v1.z - v2.z;
    return v3;
}


__device__ float dot_prod(
        fvec3 v1,
        fvec3 v2
)
{
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}


__device__ float norm_fvec3(
        fvec3 v
)
{
    return sqrt(dot_prod(v,v));
}


__device__ fvec3 operator*(
        fvec3 v,
        float f
)
{
    fvec3 o;
    o.x = v.x * f;
    o.y = v.y * f;
    o.z = v.z * f;
    return o;
}


__device__ fvec3 operator/(
        fvec3 v,
        float f
)
{
    fvec3 o;
    o.x = v.x / f;
    o.y = v.y / f;
    o.z = v.z / f;
    return o;
}


struct det_spec {
    fvec3 pos;
    fvec3 u;
    fvec3 v;
    unsigned short nu;
    unsigned short nv;
    unsigned char type;
};

struct proj_spec {
    fvec3 sourcepos;
    det_spec dspec;
};

struct vol_spec {
    unsigned long ptr_000;
    uivec3 strides_bytes;
    uivec3 shape;
    fvec3 pos_000;
};

struct vol_tex_chunk_spec {
    uivec3 offsets;
    uivec3 shape;
    fvec3 texpos;
};

struct volume_chunk_spec {
    unsigned short idx_xstart;
    unsigned short idx_xstop;
    unsigned short idx_ystart;
    unsigned short idx_ystop;
    unsigned short idx_zstart;
    unsigned short idx_zstop;
};

