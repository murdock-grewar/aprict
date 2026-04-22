struct fvec2 {
    float x;
    float y;

public:
    // implicit constructor from floats
    fvec2(float xi, float yi) : x(xi), y(yi) {}
    // default constructor, used with i.e. 'fvec2 v;'
    fvec2() = default;
};

struct uivec3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;

public:
    // implicit constructor from floats
    uivec3(unsigned int xi, unsigned int yi, unsigned int zi) : x(xi), y(yi), z(zi) {}
    // default constructor, used with i.e. 'uivec3 v;'
    uivec3() = default;
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


__device__ fvec3 operator*(
        fvec3 v1,
        fvec3 v2
)
{
    fvec3 v3;
    v3.x = v1.x * v2.x;
    v3.y = v1.y * v2.y;
    v3.z = v1.z * v2.z;
    return v3;
}


__device__ fvec3 operator/(
        fvec3 v1,
        fvec3 v2
)
{
    fvec3 v3;
    v3.x = v1.x / v2.x;
    v3.y = v1.y / v2.y;
    v3.z = v1.z / v2.z;
    return v3;
}


__device__ float dot_prod(
        fvec3 v1,
        fvec3 v2
)
{
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

__device__ fvec3 cross_prod(
        fvec3 v1,
        fvec3 v2
)
{
    return fvec3(
        v1.y*v2.z - v1.z*v2.y,
        v1.z*v2.x - v1.x*v2.z,
        v1.x*v2.y - v1.y*v2.x
    );
}

__device__ float min_of_components(
        fvec3 v
)
{
    return min(v.x, min(v.y, v.z));
}


__device__ float max_of_components(
        fvec3 v
)
{
    return max(v.x, max(v.y, v.z));
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


struct uivec2 {
    unsigned int x;
    unsigned int y;

public:
    // implicit constructor from unsigned ints
    uivec2(unsigned int xi, unsigned int yi) : x(xi), y(yi) {}
    // default constructor, used with i.e. 'uivec2 v;'
    uivec2() = default;
};

struct ulvec3 {
    unsigned long x;
    unsigned long y;
    unsigned long z;

public:
    // implicit constructor
    ulvec3(unsigned long xi, unsigned long yi, unsigned long zi) : x(xi), y(yi), z(zi) {}
    // default constructor, used with i.e. 'ulvec3 v;'
    ulvec3() = default;
};

