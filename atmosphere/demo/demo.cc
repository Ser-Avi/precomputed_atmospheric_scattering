// Avi code to dynamically move Sun.
void UpdateSunLocation(double& zenith, double& azimuth)
{
    zenith += 0.000001;
    azimuth += 0.000001;
}

#include <cstring>

const char* fullfrag_1 = R"(
       #version 330

// DEFINITIONS
#define TEMPLATE(x)
#define TEMPLATE_ARGUMENT(x)
#define assert(x)

const int TRANSMITTANCE_TEXTURE_WIDTH = 256;
const int TRANSMITTANCE_TEXTURE_HEIGHT = 64;
const int SCATTERING_TEXTURE_R_SIZE = 32;
const int SCATTERING_TEXTURE_MU_SIZE = 128;
const int SCATTERING_TEXTURE_MU_S_SIZE = 32;
const int SCATTERING_TEXTURE_NU_SIZE = 8;
const int IRRADIANCE_TEXTURE_WIDTH = 64;
const int IRRADIANCE_TEXTURE_HEIGHT = 16;

const float m = 1.0;
const float nm = 1.0;
const float rad = 1.0;
const float sr = 1.0;
const float watt = 1.0;
const float lm = 1.0;
const float PI = 3.14159265358979323846;
const float km = 1000.0 * m;
const float m2 = m * m;
const float m3 = m * m * m;
const float pi = PI * rad;
const float deg = pi / 180.0;
const float watt_per_square_meter = watt / m2;
const float watt_per_square_meter_per_sr = watt / (m2 * sr);
const float watt_per_square_meter_per_nm = watt / (m2 * nm);
const float watt_per_square_meter_per_sr_per_nm = watt / (m2 * sr * nm);
const float watt_per_cubic_meter_per_sr_per_nm = watt / (m3 * sr * nm);
const float cd = lm / sr;
const float kcd = 1000.0 * cd;
const float cd_per_square_meter = cd / m2;
const float kcd_per_square_meter = kcd / m2;

struct DensityProfileLayer {
  float width;
  float exp_term;
  float exp_scale;
  float linear_term;
  float constant_term;
};

struct DensityProfile {
  DensityProfileLayer layers[2];
};

struct AtmosphereParameters {
  vec3 solar_irradiance;
  float sun_angular_radius;
  float bottom_radius;
  float top_radius;
  DensityProfile rayleigh_density;
  vec3 rayleigh_scattering;
  DensityProfile mie_density;
  vec3 mie_scattering;
  vec3 mie_extinction;
  float mie_phase_function_g;
  DensityProfile absorption_density;
  vec3 absorption_extinction;
  vec3 ground_albedo;
  float mu_s_min;
};

const AtmosphereParameters ATMOSPHERE = AtmosphereParameters(
vec3(1.474000,1.850400,1.911980),
0.004675,
6360.000000,
6420.000000,
DensityProfile(DensityProfileLayer[2](DensityProfileLayer(0.000000,0.000000,0.000000,0.000000,0.000000),DensityProfileLayer(0.000000,1.000000,-0.125000,0.000000,0.000000))),
vec3(0.005802,0.013558,0.033100),
DensityProfile(DensityProfileLayer[2](DensityProfileLayer(0.000000,0.000000,0.000000,0.000000,0.000000),DensityProfileLayer(0.000000,1.000000,-0.833333,0.000000,0.000000))),
vec3(0.003996,0.003996,0.003996),
vec3(0.004440,0.004440,0.004440),
0.800000,
DensityProfile(DensityProfileLayer[2](DensityProfileLayer(25.000000,0.000000,0.000000,0.066667,-0.666667),DensityProfileLayer(0.000000,0.000000,0.000000,-0.066667,2.666667))),
vec3(0.000650,0.001881,0.000085),
vec3(0.100000,0.100000,0.100000),
-0.500000);

const vec3 SKY_SPECTRAL_RADIANCE_TO_LUMINANCE = vec3(683.000000,683.000000,683.000000);
const vec3 SUN_SPECTRAL_RADIANCE_TO_LUMINANCE = vec3(98242.786222,69954.398112,66475.012354);

const float kLengthUnitInMeters = 1000.000000;

// helper funcs
float ClampCosine(float mu) {
  return clamp(mu, float(-1.0), float(1.0));
}

float ClampDistance(float d) {
  return max(d, 0.0 * m);
}

float ClampRadius(const in AtmosphereParameters atmosphere, float r) {
  return clamp(r, atmosphere.bottom_radius, atmosphere.top_radius);
}

float SafeSqrt(float a) {
  return sqrt(max(a, 0.0 * m2));
}

uniform vec3 camera;
uniform float exposure;
uniform vec3 white_point;
uniform vec3 earth_center;
uniform vec3 sun_direction;
uniform vec2 sun_size;
in vec3 view_ray;
layout(location = 0) out vec4 color;
uniform sampler2D transmittance_texture;
uniform sampler2D irradiance_texture;
uniform sampler3D scattering_texture;
const vec3 kSphereCenter = vec3(0.0, 0.0, 1000.0) / kLengthUnitInMeters;
const float kSphereRadius = 1000.0 / kLengthUnitInMeters;
const vec3 kSphereAlbedo = vec3(0.8);
const vec3 kGroundAlbedo = vec3(0.0, 0.0, 0.04);


// added
// called by GetTransmittanceTextureUvFromRMu
// called by GetScatteringTextureUvwzFromRMuMuSNu
float DistanceToTopAtmosphereBoundary(const in AtmosphereParameters atmosphere,
    float r, float mu) {
  assert(r <= atmosphere.top_radius);
  assert(mu >= -1.0 && mu <= 1.0);
  float discriminant = r * r * (mu * mu - 1.0) +
      atmosphere.top_radius * atmosphere.top_radius;
  return ClampDistance(-r * mu + SafeSqrt(discriminant));
}

// added
// called by GetIrradianceTextureUvFromRMuS
// called by GetTransmittanceTextureUvFromRMu
// called by GetScatteringTextureUvwzFromRMuMuSNu
float GetTextureCoordFromUnitRange(float x, int texture_size) {
  return 0.5 / float(texture_size) + x * (1.0 - 1.0 / float(texture_size));
}

// added
// called by GetSkyRadianceToPoint
// called by GetSkyRadiance
float MiePhaseFunction(float g, float nu) {
  float k = 3.0 / (8.0 * PI * sr) * (1.0 - g * g) / (2.0 + g * g);
  return k * (1.0 + nu * nu) / pow(1.0 + g * g - 2.0 * g * nu, 1.5);
}

// added
// called by GetSkyRadiance
float RayleighPhaseFunction(float nu) {
  float k = 3.0 / (16.0 * PI * sr);
  return k * (1.0 + nu * nu);
}
)";

const char* fullfrag_2 = R"(
       // added
// called by GetIrradiance
vec2 GetIrradianceTextureUvFromRMuS(const in AtmosphereParameters atmosphere,
    float r, float mu_s) {
  assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
  assert(mu_s >= -1.0 && mu_s <= 1.0);
  float x_r = (r - atmosphere.bottom_radius) / (atmosphere.top_radius - atmosphere.bottom_radius);
  float x_mu_s = mu_s * 0.5 + 0.5;
  return vec2(GetTextureCoordFromUnitRange(x_mu_s, IRRADIANCE_TEXTURE_WIDTH),
              GetTextureCoordFromUnitRange(x_r, IRRADIANCE_TEXTURE_HEIGHT));
}

// added
// called by GetTransmittanceToTopAtmosphereBoundary
vec2 GetTransmittanceTextureUvFromRMu(const in AtmosphereParameters atmosphere,
    float r, float mu) {
  assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
  assert(mu >= -1.0 && mu <= 1.0);
  float H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
      atmosphere.bottom_radius * atmosphere.bottom_radius);
  float rho = SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
  float d = DistanceToTopAtmosphereBoundary(atmosphere, r, mu);
  float d_min = atmosphere.top_radius - r;
  float d_max = rho + H;
  float x_mu = (d - d_min) / (d_max - d_min);
  float x_r = rho / H;
  return vec2(GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_TEXTURE_WIDTH),
              GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_TEXTURE_HEIGHT));
}

// added
// called by GetCombinedScattering
vec4 GetScatteringTextureUvwzFromRMuMuSNu(const in AtmosphereParameters atmosphere,
    float r, float mu, float mu_s, float nu,
    bool ray_r_mu_intersects_ground) {
  assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
  assert(mu >= -1.0 && mu <= 1.0);
  assert(mu_s >= -1.0 && mu_s <= 1.0);
  assert(nu >= -1.0 && nu <= 1.0);

  float H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
      atmosphere.bottom_radius * atmosphere.bottom_radius);

  float rho = SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);

  float u_r = GetTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE);
  float r_mu = r * mu;

  float discriminant = r_mu * r_mu - r * r + atmosphere.bottom_radius * atmosphere.bottom_radius;
  float u_mu;

  if (ray_r_mu_intersects_ground) {
    float d = -r_mu - SafeSqrt(discriminant);
    float d_min = r - atmosphere.bottom_radius;
    float d_max = rho;
    u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(d_max == d_min ? 0.0 :
        (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
  } else {
    float d = -r_mu + SafeSqrt(discriminant + H * H);
    float d_min = atmosphere.top_radius - r;
    float d_max = rho + H;
    u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
        (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
  }

  float d = DistanceToTopAtmosphereBoundary(
      atmosphere, atmosphere.bottom_radius, mu_s);

  float d_min = atmosphere.top_radius - atmosphere.bottom_radius;
  float d_max = H;

  float a = (d - d_min) / (d_max - d_min);
  float D = DistanceToTopAtmosphereBoundary(
      atmosphere, atmosphere.bottom_radius, atmosphere.mu_s_min);

  float A = (D - d_min) / (d_max - d_min);
  float u_mu_s = GetTextureCoordFromUnitRange(
      max(1.0 - a / A, 0.0) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE);

  float u_nu = (nu + 1.0) / 2.0;
  return vec4(u_nu, u_mu_s, u_mu, u_r);
}

// added
// called by GetTransmittanceToSun
// called by GetTransmittance
// called by GetSkyRadiance
vec3 GetTransmittanceToTopAtmosphereBoundary(
    const in AtmosphereParameters atmosphere,
    const in sampler2D transmittance_texture,
    float r, float mu) {
  assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
  vec2 uv = GetTransmittanceTextureUvFromRMu(atmosphere, r, mu);
  return vec3(texture(transmittance_texture, uv));
}

// added
// called by GetSkyRadianceToPoint
// called by GetSkyRadiance
bool RayIntersectsGround(const in AtmosphereParameters atmosphere,
    float r, float mu) {
  assert(r >= atmosphere.bottom_radius);
  assert(mu >= -1.0 && mu <= 1.0);
  return mu < 0.0 && r * r * (mu * mu - 1.0) +
      atmosphere.bottom_radius * atmosphere.bottom_radius >= 0.0 * m2;
}

// added from other func
// called by GetSunAndSkyIrradiance
vec3 GetIrradiance(
    const in AtmosphereParameters atmosphere,
    const in sampler2D irradiance_texture,
    float r, float mu_s) {
  vec2 uv = GetIrradianceTextureUvFromRMuS(atmosphere, r, mu_s);
  return vec3(texture(irradiance_texture, uv));
}

// added
// called by GetCombinedScattering
// called by GetSkyRadianceToPoint
vec3 GetExtrapolatedSingleMieScattering(
    const in AtmosphereParameters atmosphere, const in vec4 scattering) {
  if (scattering.r <= 0.0) {
    return vec3(0.0);
  }
  return scattering.rgb * scattering.a / scattering.r *
	    (atmosphere.rayleigh_scattering.r / atmosphere.mie_scattering.r) *
	    (atmosphere.mie_scattering / atmosphere.rayleigh_scattering);
}

// added
// called by GetSkyRadianceToPoint
// called by GetSkyRadiance
vec3 GetTransmittance(
    const in AtmosphereParameters atmosphere,
    const in sampler2D transmittance_texture,
    float r, float mu, float d, bool ray_r_mu_intersects_ground) {
  assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
  assert(mu >= -1.0 && mu <= 1.0);
  assert(d >= 0.0 * m);

  float r_d = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
  float mu_d = ClampCosine((r * mu + d) / r_d);

  if (ray_r_mu_intersects_ground) {
    return min(
        GetTransmittanceToTopAtmosphereBoundary(
            atmosphere, transmittance_texture, r_d, -mu_d) /
        GetTransmittanceToTopAtmosphereBoundary(
            atmosphere, transmittance_texture, r, -mu),
        vec3(1.0));
  } else {
    return min(
        GetTransmittanceToTopAtmosphereBoundary(
            atmosphere, transmittance_texture, r, mu) /
        GetTransmittanceToTopAtmosphereBoundary(
            atmosphere, transmittance_texture, r_d, mu_d),
        vec3(1.0));
  }
}

// added
// called by GetSunAndSkyIrradiance
vec3 GetTransmittanceToSun(
    const in AtmosphereParameters atmosphere,
    const in sampler2D transmittance_texture,
    float r, float mu_s) {
  float sin_theta_h = atmosphere.bottom_radius / r;
  float cos_theta_h = -sqrt(max(1.0 - sin_theta_h * sin_theta_h, 0.0));
  return GetTransmittanceToTopAtmosphereBoundary(
          atmosphere, transmittance_texture, r, mu_s) *
      smoothstep(-sin_theta_h * atmosphere.sun_angular_radius / rad,
                 sin_theta_h * atmosphere.sun_angular_radius / rad,
                 mu_s - cos_theta_h);
}

// added from other func
// called by GetSunAndSkyIlluminance
vec3 GetSunAndSkyIrradiance(
    const in AtmosphereParameters atmosphere,
    const in sampler2D transmittance_texture,
    const in sampler2D irradiance_texture,
    const in vec3 point, const in vec3 normal, const in vec3 sun_direction,
    out vec3 sky_irradiance) {
  float r = length(point);
  float mu_s = dot(point, sun_direction) / r;
  sky_irradiance = GetIrradiance(atmosphere, irradiance_texture, r, mu_s) *
        (1.0 + dot(normal, point) / r) * 0.5;
  return atmosphere.solar_irradiance * GetTransmittanceToSun( atmosphere, transmittance_texture, r, mu_s) *
        max(dot(normal, sun_direction), 0.0);
}

// added
// called by GetSkyRadianceToPoint
// called by GetSkyRadiance
vec3 GetCombinedScattering(
    const in AtmosphereParameters atmosphere,
    const in sampler3D scattering_texture,
    float r, float mu, float mu_s, float nu,
    bool ray_r_mu_intersects_ground,
    out vec3 single_mie_scattering) {

  vec4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(
      atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
  float tex_coord_x = uvwz.x * float(SCATTERING_TEXTURE_NU_SIZE - 1);
  float tex_x = floor(tex_coord_x);
  float lerp = tex_coord_x - tex_x;

  vec3 uvw0 = vec3((tex_x + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE),
      uvwz.z, uvwz.w);

  vec3 uvw1 = vec3((tex_x + 1.0 + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE),
      uvwz.z, uvwz.w);

  vec4 combined_scattering =
      texture(scattering_texture, uvw0) * (1.0 - lerp) +
      texture(scattering_texture, uvw1) * lerp;

  vec3 scattering = vec3(combined_scattering);

  single_mie_scattering = GetExtrapolatedSingleMieScattering(atmosphere, combined_scattering);
  return scattering;

}

// added
// called by GetSkyLuminance
vec3 GetSkyRadiance(
    const in AtmosphereParameters atmosphere,
    const in sampler2D transmittance_texture,
    const in sampler3D scattering_texture,
    vec3 camera, const in vec3 view_ray, float shadow_length,
    const in vec3 sun_direction, out vec3 transmittance) {
  float r = length(camera);
  float rmu = dot(camera, view_ray);
  float distance_to_top_atmosphere_boundary = -rmu -
      sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);

  if (distance_to_top_atmosphere_boundary > 0.0 * m) {
    camera = camera + view_ray * distance_to_top_atmosphere_boundary;
    r = atmosphere.top_radius;
    rmu += distance_to_top_atmosphere_boundary;
  } else if (r > atmosphere.top_radius) {
    transmittance = vec3(1.0);
    return vec3(0.0 * watt_per_square_meter_per_sr_per_nm);
  }

  float mu = rmu / r;
  float mu_s = dot(camera, sun_direction) / r;
  float nu = dot(view_ray, sun_direction);

  bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);

  transmittance = ray_r_mu_intersects_ground ? vec3(0.0) :
      GetTransmittanceToTopAtmosphereBoundary(
          atmosphere, transmittance_texture, r, mu);

  vec3 single_mie_scattering;
  vec3 scattering;

  if (shadow_length == 0.0 * m) {
    scattering = GetCombinedScattering(
        atmosphere, scattering_texture,
        r, mu, mu_s, nu, ray_r_mu_intersects_ground,
        single_mie_scattering);
  } else {
    float d = shadow_length;
    float r_p = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
    float mu_p = (r * mu + d) / r_p;
    float mu_s_p = (r * mu_s + d * nu) / r_p;

    scattering = GetCombinedScattering( atmosphere, scattering_texture,
        r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,
        single_mie_scattering);

    vec3 shadow_transmittance =
        GetTransmittance(atmosphere, transmittance_texture,
            r, mu, shadow_length, ray_r_mu_intersects_ground);

    scattering = scattering * shadow_transmittance;
    single_mie_scattering = single_mie_scattering * shadow_transmittance;
  }
  return scattering * RayleighPhaseFunction(nu) + single_mie_scattering *
      MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
}

// added
// called by GetSkyLuminanceToPoint
vec3 GetSkyRadianceToPoint(
    const in AtmosphereParameters atmosphere,
    const in sampler2D transmittance_texture,
    const in sampler3D scattering_texture,
    vec3 camera, const in vec3 point, float shadow_length,
    const in vec3 sun_direction, out vec3 transmittance) {
  vec3 view_ray = normalize(point - camera);
  float r = length(camera);
  float rmu = dot(camera, view_ray);
  float distance_to_top_atmosphere_boundary = -rmu - sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);
  if (distance_to_top_atmosphere_boundary > 0.0 * m) {
    camera = camera + view_ray * distance_to_top_atmosphere_boundary;
    r = atmosphere.top_radius;
    rmu += distance_to_top_atmosphere_boundary;
  }
  float mu = rmu / r;
  float mu_s = dot(camera, sun_direction) / r;
  float nu = dot(view_ray, sun_direction);
  float d = length(point - camera);
  bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);

  // PR FIX
  if (!ray_r_mu_intersects_ground) {
      float mu_horiz = -SafeSqrt(1.0 - (atmosphere.bottom_radius / r) * (atmosphere.bottom_radius / r));
      mu = max(mu, mu_horiz + 0.004);
  }

  transmittance = GetTransmittance(atmosphere, transmittance_texture,
      r, mu, d, ray_r_mu_intersects_ground);
    
  vec3 single_mie_scattering;
  vec3 scattering = GetCombinedScattering(
      atmosphere, scattering_texture,
      r, mu, mu_s, nu, ray_r_mu_intersects_ground,
      single_mie_scattering);

  d = max(d - shadow_length, 0.0 * m);
  float r_p = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
  float mu_p = (r * mu + d) / r_p;
  float mu_s_p = (r * mu_s + d * nu) / r_p;

  vec3 single_mie_scattering_p;
  vec3 scattering_p = GetCombinedScattering(
      atmosphere, scattering_texture,
      r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,
      single_mie_scattering_p);

  vec3 shadow_transmittance = transmittance;

  if (shadow_length > 0.0 * m) {
    shadow_transmittance = GetTransmittance(atmosphere, transmittance_texture,
        r, mu, d, ray_r_mu_intersects_ground);
  }

  scattering = scattering - shadow_transmittance * scattering_p;

  single_mie_scattering = single_mie_scattering - shadow_transmittance * single_mie_scattering_p;
  single_mie_scattering = GetExtrapolatedSingleMieScattering(
      atmosphere, vec4(scattering, single_mie_scattering.r));

  single_mie_scattering = single_mie_scattering * smoothstep(float(0.0), float(0.01), mu_s);

  return scattering * RayleighPhaseFunction(nu) + single_mie_scattering *
      MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
}
)";

const char* fullfrag_3 = R"(
// expanded
vec3 GetSolarLuminance(){
    return ATMOSPHERE.solar_irradiance /
        (PI * ATMOSPHERE.sun_angular_radius * ATMOSPHERE.sun_angular_radius) *
        SUN_SPECTRAL_RADIANCE_TO_LUMINANCE;
}

// expanded
vec3 GetSkyLuminance(vec3 camera, vec3 view_ray, float shadow_length,
    vec3 sun_direction, out vec3 transmittance) {
  return GetSkyRadiance(ATMOSPHERE, transmittance_texture,
      scattering_texture,
      camera, view_ray, shadow_length, sun_direction, transmittance) *
      SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
}

// expanded
vec3 GetSkyLuminanceToPoint(
    vec3 camera, vec3 point, float shadow_length,
    vec3 sun_direction, out vec3 transmittance) {
  return GetSkyRadianceToPoint(ATMOSPHERE, transmittance_texture,
      scattering_texture, camera, point, shadow_length, sun_direction, transmittance) *
      SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
}

// expanded
vec3 GetSunAndSkyIlluminance(vec3 p, vec3 normal, vec3 sun_direction,
   out vec3 sky_irradiance) {
  vec3 sun_irradiance = GetSunAndSkyIrradiance(
      ATMOSPHERE, transmittance_texture, irradiance_texture, p, normal,
      sun_direction, sky_irradiance);
  sky_irradiance *= SKY_SPECTRAL_RADIANCE_TO_LUMINANCE;
  return sun_irradiance * SUN_SPECTRAL_RADIANCE_TO_LUMINANCE;
}
float GetSunVisibility(vec3 point, vec3 sun_direction) {
  vec3 p = point - kSphereCenter;
  float p_dot_v = dot(p, sun_direction);
  float p_dot_p = dot(p, p);
  float ray_sphere_center_squared_distance = p_dot_p - p_dot_v * p_dot_v;
  float discriminant =
      kSphereRadius * kSphereRadius - ray_sphere_center_squared_distance;
  if (discriminant >= 0.0) {
    float distance_to_intersection = -p_dot_v - sqrt(discriminant);
    if (distance_to_intersection > 0.0) {
      float ray_sphere_distance =
          kSphereRadius - sqrt(ray_sphere_center_squared_distance);
      float ray_sphere_angular_distance = -ray_sphere_distance / p_dot_v;
      return smoothstep(1.0, 0.0, ray_sphere_angular_distance / sun_size.x);
    }
  }
  return 1.0;
}
float GetSkyVisibility(vec3 point) {
  vec3 p = point - kSphereCenter;
  float p_dot_p = dot(p, p);
  return
      1.0 + p.z / sqrt(p_dot_p) * kSphereRadius * kSphereRadius / p_dot_p;
}
void GetSphereShadowInOut(vec3 view_direction, vec3 sun_direction,
    out float d_in, out float d_out) {
  vec3 pos = camera - kSphereCenter;
  float pos_dot_sun = dot(pos, sun_direction);
  float view_dot_sun = dot(view_direction, sun_direction);
  float k = sun_size.x;
  float l = 1.0 + k * k;
  float a = 1.0 - l * view_dot_sun * view_dot_sun;
  float b = dot(pos, view_direction) - l * pos_dot_sun * view_dot_sun -
      k * kSphereRadius * view_dot_sun;
  float c = dot(pos, pos) - l * pos_dot_sun * pos_dot_sun -
      2.0 * k * kSphereRadius * pos_dot_sun - kSphereRadius * kSphereRadius;
  float discriminant = b * b - a * c;
  if (discriminant > 0.0) {
    d_in = max(0.0, (-b - sqrt(discriminant)) / a);
    d_out = (-b + sqrt(discriminant)) / a;
    float d_base = -pos_dot_sun / view_dot_sun;
    float d_apex = -(pos_dot_sun + kSphereRadius / k) / view_dot_sun;
    if (view_dot_sun > 0.0) {
      d_in = max(d_in, d_apex);
      d_out = a > 0.0 ? min(d_out, d_base) : d_base;
    } else {
      d_in = a > 0.0 ? max(d_in, d_base) : d_base;
      d_out = min(d_out, d_apex);
    }
  } else {
    d_in = 0.0;
    d_out = 0.0;
  }
}


void main() {
    vec3 view_direction = normalize(view_ray);
    float fragment_angular_size =
        length(dFdx(view_ray) + dFdy(view_ray)) / length(view_ray);
    float shadow_in;
    float shadow_out;
    GetSphereShadowInOut(view_direction, sun_direction, shadow_in, shadow_out);
    float lightshaft_fadein_hack = smoothstep(
        0.02, 0.04, dot(normalize(camera - earth_center), sun_direction));
    /*
    <p>We then test whether the view ray intersects the sphere S or not. If it does,
    we compute an approximate (and biased) opacity value, using the same
    approximation as in <code>GetSunVisibility</code>:
    */
    vec3 p = camera - kSphereCenter;
    float p_dot_v = dot(p, view_direction);
    float p_dot_p = dot(p, p);
    float ray_sphere_center_squared_distance = p_dot_p - p_dot_v * p_dot_v;
    float discriminant =
        kSphereRadius * kSphereRadius - ray_sphere_center_squared_distance;
    float sphere_alpha = 0.0;
    vec3 sphere_radiance = vec3(0.0);
    if (discriminant >= 0.0) {
        float distance_to_intersection = -p_dot_v - sqrt(discriminant);
        if (distance_to_intersection > 0.0) {
            float ray_sphere_distance =
                kSphereRadius - sqrt(ray_sphere_center_squared_distance);
            float ray_sphere_angular_distance = -ray_sphere_distance / p_dot_v;
            sphere_alpha =
                min(ray_sphere_angular_distance / fragment_angular_size, 1.0);
            /*
            <p>We can then compute the intersection point and its normal, and use them to
            get the sun and sky irradiance received at this point. The reflected radiance
            follows, by multiplying the irradiance with the sphere BRDF:
            */
            vec3 point = camera + view_direction * distance_to_intersection;
            vec3 normal = normalize(point - kSphereCenter);
            vec3 sky_irradiance;
            vec3 sun_irradiance = GetSunAndSkyIlluminance(
                point - earth_center, normal, sun_direction, sky_irradiance);
            sphere_radiance =
                kSphereAlbedo * (1.0 / PI) * (sun_irradiance + sky_irradiance);
            /*
            <p>Finally, we take into account the aerial perspective between the camera and
            the sphere, which depends on the length of this segment which is in shadow:
            */
            float shadow_length =
                max(0.0, min(shadow_out, distance_to_intersection) - shadow_in) *
                lightshaft_fadein_hack;
            vec3 transmittance;
            vec3 in_scatter = GetSkyLuminanceToPoint(camera - earth_center,
                point - earth_center, shadow_length, sun_direction, transmittance);
            sphere_radiance = sphere_radiance * transmittance + in_scatter;
        }
    }
    /*
    <p>In the following we repeat the same steps as above, but for the planet sphere
    P instead of the sphere S (a smooth opacity is not really needed here, so we
    don't compute it. Note also how we modulate the sun and sky irradiance received
    on the ground by the sun and sky visibility factors):
    */
    p = camera - earth_center;
    p_dot_v = dot(p, view_direction);
    p_dot_p = dot(p, p);
    float ray_earth_center_squared_distance = p_dot_p - p_dot_v * p_dot_v;
    discriminant =
        earth_center.z * earth_center.z - ray_earth_center_squared_distance;
    float ground_alpha = 0.0;
    vec3 ground_radiance = vec3(0.0);
    if (discriminant >= 0.0) {
        float distance_to_intersection = -p_dot_v - sqrt(discriminant);
        if (distance_to_intersection > 0.0) {
            vec3 point = camera + view_direction * distance_to_intersection;
            vec3 normal = normalize(point - earth_center);
            vec3 sky_irradiance;
            vec3 sun_irradiance = GetSunAndSkyIlluminance(
                point - earth_center, normal, sun_direction, sky_irradiance);
            ground_radiance = kGroundAlbedo * (1.0 / PI) * (
                sun_irradiance * GetSunVisibility(point, sun_direction) +
                sky_irradiance * GetSkyVisibility(point));
            float shadow_length =
                max(0.0, min(shadow_out, distance_to_intersection) - shadow_in) *
                lightshaft_fadein_hack;
            vec3 transmittance;
            vec3 in_scatter = GetSkyLuminanceToPoint(camera - earth_center,
                point - earth_center, shadow_length, sun_direction, transmittance);
            ground_radiance = ground_radiance * transmittance + in_scatter;
            ground_alpha = 1.0;
        }
    }
    /*
    <p>Finally, we compute the radiance and transmittance of the sky, and composite
    together, from back to front, the radiance and opacities of all the objects of
    the scene:
    */
    float shadow_length = max(0.0, shadow_out - shadow_in) *
        lightshaft_fadein_hack;
    vec3 transmittance;
    vec3 radiance = GetSkyLuminance(
        camera - earth_center, view_direction, shadow_length, sun_direction,
        transmittance);
    if (dot(view_direction, sun_direction) > sun_size.y) {
        radiance = radiance + transmittance * GetSolarLuminance();
    }
    radiance = mix(radiance, ground_radiance, ground_alpha);
    radiance = mix(radiance, sphere_radiance, sphere_alpha);
    color.rgb =
        pow(vec3(1.0) - exp(-radiance / white_point * exposure), vec3(1.0 / 2.2));
    color.a = 1.0;

    //vec3 texCoord = vec3(vec2(gl_FragCoord.xy / vec2(8.f * 32.f, 128.f)),  25.f);
    //color = texture(scattering_texture, texCoord);
}
)";

/**
 * Copyright (c) 2017 Eric Bruneton
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holders nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/*<h2>atmosphere/demo/demo.cc</h2>

<p>This file, together with the shader <a href="demo.glsl.html">demo.glsl</a>,
shows how the API provided in <a href="../model.h.html">model.h</a> can be used
in practice. It implements the <code>Demo</code> class whose header is defined
in <a href="demo.h.html">demo.h</a> (note that most of the following code is
independent of our atmosphere model. The only part which is related to it is the
<code>InitModel</code> method).
*/

#include "atmosphere/demo/demo.h"

#include <glad/glad.h>
#include <GL/freeglut.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace atmosphere {
namespace demo {

/*
<p>Our demo application renders a simple scene made of a purely spherical planet
and a large sphere (of 1km radius) lying on it. This scene is displayed by
rendering a full screen quad, with a fragment shader computing the color of each
pixel by "ray tracing" (which is very simple here because the scene consists of
only two spheres). The vertex shader is thus very simple, and is provided in the
following constant. The fragment shader is more complex, and is defined in the
separate file <a href="demo.glsl.html">demo.glsl</a> (which is included here as
a string literal via the generated file <code>demo.glsl.inc</code>):
*/

namespace {

constexpr double kPi = 3.1415926;
constexpr double kSunAngularRadius = 0.00935 / 2.0;
constexpr double kSunSolidAngle = kPi * kSunAngularRadius * kSunAngularRadius;
constexpr double kLengthUnitInMeters = 1000.0;

const char kVertexShader[] = R"(
    #version 330
    uniform mat4 model_from_view;
    uniform mat4 view_from_clip;
    layout(location = 0) in vec4 vertex;
    out vec3 view_ray;
    void main() {
      view_ray =
          (model_from_view * vec4((view_from_clip * vertex).xyz, 0.0)).xyz;
      gl_Position = vertex;
    })";

//#include "atmosphere/demo/demo.glsl.inc"

static std::map<int, Demo*> INSTANCES;

}  // anonymous namespace

/*
<p>The class constructor is straightforward and completely independent of our
atmosphere model (which is initialized in the separate method
<code>InitModel</code>). It's main role is to create the demo window, to set up
the event handler callbacks (it does so in such a way that several Demo
instances can be created at the same time, using the <code>INSTANCES</code>
global variable), and to create the vertex buffers and the text renderer that
will be used to render the scene and the help messages:
*/

Demo::Demo(int viewport_width, int viewport_height) :
    use_constant_solar_spectrum_(false),
    use_ozone_(true),
    use_combined_textures_(true),
    use_half_precision_(false),
    use_luminance_(PRECOMPUTED),
    do_white_balance_(true),
    show_help_(true),
    program_(0),
    view_distance_meters_(9000.0),
    view_zenith_angle_radians_(1.47),
    view_azimuth_angle_radians_(-0.1),
    sun_zenith_angle_radians_(1.3),
    sun_azimuth_angle_radians_(2.9),
    display_texture_(0), // avi added
    scatter_slice_(31),
    exposure_(10.0) {
  glutInitWindowSize(viewport_width, viewport_height);
  window_id_ = glutCreateWindow("Atmosphere Demo");
  INSTANCES[window_id_] = this;
  if (!gladLoadGL()) {
    throw std::runtime_error("GLAD initialization failed");
  }
  if (!GLAD_GL_VERSION_3_3) {
    throw std::runtime_error("OpenGL 3.3 or higher is required");
  }

  glutDisplayFunc([]() {
    INSTANCES[glutGetWindow()]->HandleRedisplayEvent();
  });
  glutReshapeFunc([](int width, int height) {
    INSTANCES[glutGetWindow()]->HandleReshapeEvent(width, height);
  });
  glutKeyboardFunc([](unsigned char key, int x, int y) {
    INSTANCES[glutGetWindow()]->HandleKeyboardEvent(key);
  });
  glutMouseFunc([](int button, int state, int x, int y) {
    INSTANCES[glutGetWindow()]->HandleMouseClickEvent(button, state, x, y);
  });
  glutMotionFunc([](int x, int y) {
    INSTANCES[glutGetWindow()]->HandleMouseDragEvent(x, y);
  });
  glutMouseWheelFunc([](int button, int dir, int x, int y) {
    INSTANCES[glutGetWindow()]->HandleMouseWheelEvent(dir);
  });

  glGenVertexArrays(1, &full_screen_quad_vao_);
  glBindVertexArray(full_screen_quad_vao_);
  glGenBuffers(1, &full_screen_quad_vbo_);
  glBindBuffer(GL_ARRAY_BUFFER, full_screen_quad_vbo_);
  const GLfloat vertices[] = {
    -1.0, -1.0, 0.0, 1.0,
    +1.0, -1.0, 0.0, 1.0,
    -1.0, +1.0, 0.0, 1.0,
    +1.0, +1.0, 0.0, 1.0,
  };
  glBufferData(GL_ARRAY_BUFFER, sizeof vertices, vertices, GL_STATIC_DRAW);
  constexpr GLuint kAttribIndex = 0;
  constexpr int kCoordsPerVertex = 4;
  glVertexAttribPointer(kAttribIndex, kCoordsPerVertex, GL_FLOAT, false, 0, 0);
  glEnableVertexAttribArray(kAttribIndex);
  glBindVertexArray(0);

  text_renderer_.reset(new TextRenderer);

  InitModel();
}

/*
<p>The destructor is even simpler:
*/

Demo::~Demo() {
  glDeleteShader(vertex_shader_);
  glDeleteShader(fragment_shader_);
  glDeleteProgram(program_);
  glDeleteBuffers(1, &full_screen_quad_vbo_);
  glDeleteVertexArrays(1, &full_screen_quad_vao_);
  INSTANCES.erase(window_id_);
}

/*
<p>The "real" initialization work, which is specific to our atmosphere model,
is done in the following method. It starts with the creation of an atmosphere
<code>Model</code> instance, with parameters corresponding to the Earth
atmosphere:
*/

void Demo::InitModel() {
  // Values from "Reference Solar Spectral Irradiance: ASTM G-173", ETR column
  // (see http://rredc.nrel.gov/solar/spectra/am1.5/ASTMG173/ASTMG173.html),
  // summed and averaged in each bin (e.g. the value for 360nm is the average
  // of the ASTM G-173 values for all wavelengths between 360 and 370nm).
  // Values in W.m^-2 per nanometer.
  constexpr int kLambdaMin = 360;
  constexpr int kLambdaMax = 830;
  constexpr double kSolarIrradiance[48] = {
    1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054, 1.6887, 1.61253,
    1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298,
    1.8685, 1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158, 1.7533,
    1.6965, 1.68194, 1.64654, 1.6048, 1.52143, 1.55622, 1.5113, 1.474, 1.4482,
    1.41018, 1.36775, 1.34188, 1.31429, 1.28303, 1.26758, 1.2367, 1.2082,
    1.18737, 1.14683, 1.12362, 1.1058, 1.07124, 1.04992
  };
  // Values from http://www.iup.uni-bremen.de/gruppen/molspec/databases/
  // referencespectra/o3spectra2011/index.html for 233K, summed and averaged in
  // each bin (e.g. the value for 360nm is the average of the original values
  // for all wavelengths between 360 and 370nm). Values in m^2.
  constexpr double kOzoneCrossSection[48] = {
    1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27,
    8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26,
    1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25,
    4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25,
    2.662e-25, 2.238e-25, 1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26,
    6.566e-26, 5.105e-26, 4.15e-26, 4.228e-26, 3.237e-26, 2.451e-26, 2.801e-26,
    2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27
  };
  // From https://en.wikipedia.org/wiki/Dobson_unit, in molecules.m^-2.
  constexpr double kDobsonUnit = 2.687e20;
  // Maximum number density of ozone molecules, in m^-3 (computed so at to get
  // 300 Dobson units of ozone - for this we divide 300 DU by the integral of
  // the ozone density profile defined below, which is equal to 15km).
  constexpr double kMaxOzoneNumberDensity = 300.0 * kDobsonUnit / 15000.0;
  // Wavelength independent solar irradiance "spectrum" (not physically
  // realistic, but was used in the original implementation).
  constexpr double kConstantSolarIrradiance = 1.5;
  constexpr double kBottomRadius = 6360000.0;
  constexpr double kTopRadius = 6420000.0;
  constexpr double kRayleigh = 1.24062e-6;
  constexpr double kRayleighScaleHeight = 8000.0;
  constexpr double kMieScaleHeight = 1200.0;
  constexpr double kMieAngstromAlpha = 0.0;
  constexpr double kMieAngstromBeta = 5.328e-3;
  constexpr double kMieSingleScatteringAlbedo = 0.9;
  constexpr double kMiePhaseFunctionG = 0.8;
  constexpr double kGroundAlbedo = 0.1;
  const double max_sun_zenith_angle =
      (120.0) / 180.0 * kPi;      // always full precision

  DensityProfileLayer
      rayleigh_layer(0.0, 1.0, -1.0 / kRayleighScaleHeight, 0.0, 0.0);
  DensityProfileLayer mie_layer(0.0, 1.0, -1.0 / kMieScaleHeight, 0.0, 0.0);
  // Density profile increasing linearly from 0 to 1 between 10 and 25km, and
  // decreasing linearly from 1 to 0 between 25 and 40km. This is an approximate
  // profile from http://www.kln.ac.lk/science/Chemistry/Teaching_Resources/
  // Documents/Introduction%20to%20atmospheric%20chemistry.pdf (page 10).
  std::vector<DensityProfileLayer> ozone_density;
  ozone_density.push_back(
      DensityProfileLayer(25000.0, 0.0, 0.0, 1.0 / 15000.0, -2.0 / 3.0));
  ozone_density.push_back(
      DensityProfileLayer(0.0, 0.0, 0.0, -1.0 / 15000.0, 8.0 / 3.0));

  std::vector<double> wavelengths;
  std::vector<double> solar_irradiance;
  std::vector<double> rayleigh_scattering;
  std::vector<double> mie_scattering;
  std::vector<double> mie_extinction;
  std::vector<double> absorption_extinction;
  std::vector<double> ground_albedo;
  for (int l = kLambdaMin; l <= kLambdaMax; l += 10) {
    double lambda = static_cast<double>(l) * 1e-3;  // micro-meters
    double mie =
        kMieAngstromBeta / kMieScaleHeight * pow(lambda, -kMieAngstromAlpha);
    wavelengths.push_back(l);
    solar_irradiance.push_back(kSolarIrradiance[(l - kLambdaMin) / 10]);      // we are always using realistic solar spectrum
    rayleigh_scattering.push_back(kRayleigh * pow(lambda, -4));
    mie_scattering.push_back(mie * kMieSingleScatteringAlbedo);
    mie_extinction.push_back(mie);
    absorption_extinction.push_back(kMaxOzoneNumberDensity * kOzoneCrossSection[(l - kLambdaMin) / 10]);        // we are always using ozone
    ground_albedo.push_back(kGroundAlbedo);
  }

  model_.reset(new Model(wavelengths, solar_irradiance, kSunAngularRadius,
      kBottomRadius, kTopRadius, {rayleigh_layer}, rayleigh_scattering,
      {mie_layer}, mie_scattering, mie_extinction, kMiePhaseFunctionG,
      ozone_density, absorption_extinction, ground_albedo, max_sun_zenith_angle,
      kLengthUnitInMeters, 15,       // we are always precomputed
      true, false));            // always combined, full precision
  //model_->Init();

/*
<p>Then, it creates and compiles the vertex and fragment shaders used to render
our demo scene, and link them with the <code>Model</code>'s atmosphere shader
to get the final scene rendering program:
*/

  vertex_shader_ = glCreateShader(GL_VERTEX_SHADER);
  const char* const vertex_shader_source = kVertexShader;
  glShaderSource(vertex_shader_, 1, &vertex_shader_source, NULL);
  glCompileShader(vertex_shader_);

  //const std::string fragment_shader_str =
  //    "#version 330\n" +
  //    std::string("#define USE_LUMINANCE\n") +        //always using luminance
  //    "const float kLengthUnitInMeters = " +
  //    std::to_string(kLengthUnitInMeters) + ";\n" +
  //    demo_glsl;
  //const char* fragment_shader_source = fragment_shader_str.c_str();

  const char* fullfrag_[] = { fullfrag_1, fullfrag_2, fullfrag_3 };



  fragment_shader_ = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader_, 3, fullfrag_, NULL);
  //glShaderSource(fragment_shader_, 1, &fragment_shader_source, NULL);
  glCompileShader(fragment_shader_);

  if (program_ != 0) {
    glDeleteProgram(program_);
  }
  program_ = glCreateProgram();
  glAttachShader(program_, vertex_shader_);
  glAttachShader(program_, fragment_shader_);
  //glAttachShader(program_, model_->shader());       // comment this out
  glLinkProgram(program_);
  glDetachShader(program_, vertex_shader_);
  glDetachShader(program_, fragment_shader_);
  //glDetachShader(program_, model_->shader());       // this

/*
<p>Finally, it sets the uniforms of this program that can be set once and for
all (in our case this includes the <code>Model</code>'s texture uniforms,
because our demo app does not have any texture of its own):
*/

  glUseProgram(program_);
  model_->SetProgramUniforms(program_, 0, 1, 2, 3);
  double white_point_r = 1.0;
  double white_point_g = 1.0;
  double white_point_b = 1.0;
    Model::ConvertSpectrumToLinearSrgb(wavelengths, solar_irradiance,
    &white_point_r, &white_point_g, &white_point_b);
    double white_point = (white_point_r + white_point_g + white_point_b) / 3.0;
    white_point_r /= white_point;
    white_point_g /= white_point;
    white_point_b /= white_point;   // always using white balance
    printf("\nWHITE POINT: %lf, %lf, %lf", white_point_r, white_point_g, white_point_b);
  glUniform3f(glGetUniformLocation(program_, "white_point"),
      white_point_r, white_point_g, white_point_b);
  glUniform3f(glGetUniformLocation(program_, "earth_center"),
      0.0, 0.0, -kBottomRadius / kLengthUnitInMeters);
  printf("\nEarth center: %lf, %lf, %lf", 0.0, 0.0, -kBottomRadius / kLengthUnitInMeters);
  glUniform2f(glGetUniformLocation(program_, "sun_size"),
      tan(kSunAngularRadius),
      cos(kSunAngularRadius));
  printf("\nSUN Size: %lf, %lf\n", tan(kSunAngularRadius), cos(kSunAngularRadius));

  glUniform1i(glGetUniformLocation(program_, "display_texture"), display_texture_);
  glUniform1i(glGetUniformLocation(program_, "scatter_slice"), scatter_slice_);

  // This sets 'view_from_clip', which only depends on the window size.
  HandleReshapeEvent(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
}

/*
<p>The scene rendering method simply sets the uniforms related to the camera
position and to the Sun direction, and then draws a full screen quad (and
optionally a help screen).
*/

void Demo::HandleRedisplayEvent() {
    // Avi added
    //UpdateSunLocation(sun_zenith_angle_radians_, sun_azimuth_angle_radians_);

  // Unit vectors of the camera frame, expressed in world space.
  float cos_z = cos(view_zenith_angle_radians_);
  float sin_z = sin(view_zenith_angle_radians_);
  float cos_a = cos(view_azimuth_angle_radians_);
  float sin_a = sin(view_azimuth_angle_radians_);
  float ux[3] = { -sin_a, cos_a, 0.0 };
  float uy[3] = { -cos_z * cos_a, -cos_z * sin_a, sin_z };
  float uz[3] = { sin_z * cos_a, sin_z * sin_a, cos_z };
  float l = view_distance_meters_ / kLengthUnitInMeters;

  // Transform matrix from camera frame to world space (i.e. the inverse of a
  // GL_MODELVIEW matrix).
  float model_from_view[16] = {
    ux[0], uy[0], uz[0], uz[0] * l,
    ux[1], uy[1], uz[1], uz[1] * l,
    ux[2], uy[2], uz[2], uz[2] * l,
    0.0, 0.0, 0.0, 1.0
  };

  glUniform3f(glGetUniformLocation(program_, "camera"),
      model_from_view[3],
      model_from_view[7],
      model_from_view[11]);
  glUniform1f(glGetUniformLocation(program_, "exposure"), exposure_ * 1e-5);    // always use luminance
  glUniformMatrix4fv(glGetUniformLocation(program_, "model_from_view"),
      1, true, model_from_view);
  glUniform3f(glGetUniformLocation(program_, "sun_direction"),
      cos(sun_azimuth_angle_radians_) * sin(sun_zenith_angle_radians_),
      sin(sun_azimuth_angle_radians_) * sin(sun_zenith_angle_radians_),
      cos(sun_zenith_angle_radians_));
  //printf("\nSUN DIR: %lf, %lf, %lf", cos(sun_azimuth_angle_radians_) * sin(sun_zenith_angle_radians_), sin(sun_azimuth_angle_radians_) * sin(sun_zenith_angle_radians_), cos(sun_zenith_angle_radians_));

  glBindVertexArray(full_screen_quad_vao_);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glBindVertexArray(0);

  if (false) {      // this is show_help if we want it
    std::stringstream help;
    help << "Mouse:\n"
         << " drag, CTRL+drag, wheel: view and sun directions\n"
         << "Keys:\n"
         << " h: help\n"
         << " s: solar spectrum (currently: "
         << (use_constant_solar_spectrum_ ? "constant" : "realistic") << ")\n"
         << " o: ozone (currently: " << (use_ozone_ ? "on" : "off") << ")\n"
         << " t: combine textures (currently: "
         << (use_combined_textures_ ? "on" : "off") << ")\n"
         << " p: half precision (currently: "
         << (use_half_precision_ ? "on" : "off") << ")\n"
         << " l: use luminance (currently: "
         << (use_luminance_ == PRECOMPUTED ? "precomputed" :
             (use_luminance_ == APPROXIMATE ? "approximate" : "off")) << ")\n"
         << " w: white balance (currently: "
         << (do_white_balance_ ? "on" : "off") << ")\n"
         << " +/-: increase/decrease exposure (" << exposure_ << ")\n"
         << " 1-9: predefined views\n"
         << "s: scatter texture slice(0-31): (" << scatter_slice_ << ")\n";
    text_renderer_->SetColor(1.0, 0.0, 0.0);
    text_renderer_->DrawText(help.str(), 5, 4);
  }

  glutSwapBuffers();
  glutPostRedisplay();
}

/*
<p>The other event handling methods are also straightforward, and do not
interact with the atmosphere model:
*/

void Demo::HandleReshapeEvent(int viewport_width, int viewport_height) {
  glViewport(0, 0, viewport_width, viewport_height);

  const float kFovY = 50.0 / 180.0 * kPi;
  const float kTanFovY = tan(kFovY / 2.0);
  float aspect_ratio = static_cast<float>(viewport_width) / viewport_height;

  // Transform matrix from clip space to camera space (i.e. the inverse of a
  // GL_PROJECTION matrix).
  float view_from_clip[16] = {
    kTanFovY * aspect_ratio, 0.0, 0.0, 0.0,
    0.0, kTanFovY, 0.0, 0.0,
    0.0, 0.0, 0.0, -1.0,
    0.0, 0.0, 1.0, 1.0
  };
  glUniformMatrix4fv(glGetUniformLocation(program_, "view_from_clip"), 1, true,
      view_from_clip);
}

void Demo::HandleKeyboardEvent(unsigned char key) {
  if (key == 27) {
    glutDestroyWindow(window_id_);
  } else if (key == 'h') {
    show_help_ = !show_help_;
  } else if (key == 's') {
      scatter_slice_ = (scatter_slice_ + 1) % 33;
      glUseProgram(program_);
      glUniform1i(glGetUniformLocation(program_, "scatter_slice"), scatter_slice_);
    //use_constant_solar_spectrum_ = !use_constant_solar_spectrum_;
  } else if (key == 'o') {
    //use_ozone_ = !use_ozone_;
  } else if (key == 't') {
    //use_combined_textures_ = !use_combined_textures_;
  } else if (key == 'p') {
    //use_half_precision_ = !use_half_precision_;
  }
  else if (key == 'l') {
      /* switch (use_luminance_) {
         case NONE: use_luminance_ = APPROXIMATE; break;
         case APPROXIMATE: use_luminance_ = PRECOMPUTED; break;
         case PRECOMPUTED: use_luminance_ = NONE; break;
       }*/
  } else if (key == 'm') {
      display_texture_ = (display_texture_ + 1) % 4;
      glUseProgram(program_);
      glUniform1i(glGetUniformLocation(program_, "display_texture"), display_texture_);
  } else if (key == 'w') {
    //do_white_balance_ = !do_white_balance_;
  } else if (key == '+') {
    exposure_ *= 1.1;
  } else if (key == '-') {
    exposure_ /= 1.1;
  } else if (key == '1') {
    SetView(9000.0, 1.47, 0.0, 1.3, 3.0, 10.0);
  } else if (key == '2') {
    SetView(9000.0, 1.47, 0.0, 1.564, -3.0, 10.0);
  } else if (key == '3') {
    SetView(7000.0, 1.57, 0.0, 1.54, -2.96, 10.0);
  } else if (key == '4') {
    SetView(7000.0, 1.57, 0.0, 1.328, -3.044, 10.0);
  } else if (key == '5') {
    SetView(9000.0, 1.39, 0.0, 1.2, 0.7, 10.0);
  } else if (key == '6') {
    SetView(9000.0, 1.5, 0.0, 1.628, 1.05, 200.0);
  } else if (key == '7') {
    SetView(7000.0, 1.43, 0.0, 1.57, 1.34, 40.0);
  } else if (key == '8') {
    SetView(2.7e6, 0.81, 0.0, 1.57, 2.0, 10.0);
  } else if (key == '9') {
    SetView(1.2e7, 0.0, 0.0, 0.93, -2.0, 10.0);
  }
  if (key == 's' || key == 'o' || key == 't' || key == 'p' || key == 'l' ||
      key == 'w') {
    InitModel();
  }
}

void Demo::HandleMouseClickEvent(
    int button, int state, int mouse_x, int mouse_y) {
  previous_mouse_x_ = mouse_x;
  previous_mouse_y_ = mouse_y;
  is_ctrl_key_pressed_ = (glutGetModifiers() & GLUT_ACTIVE_CTRL) != 0;

  if ((button == 3) || (button == 4)) {
    if (state == GLUT_DOWN) {
      HandleMouseWheelEvent(button == 3 ? 1 : -1);
    }
  }
}

void Demo::HandleMouseDragEvent(int mouse_x, int mouse_y) {
  constexpr double kScale = 500.0;
  if (is_ctrl_key_pressed_) {
    sun_zenith_angle_radians_ -= (previous_mouse_y_ - mouse_y) / kScale;
    sun_zenith_angle_radians_ =
        std::max(0.0, std::min(kPi, sun_zenith_angle_radians_));
    sun_azimuth_angle_radians_ += (previous_mouse_x_ - mouse_x) / kScale;
  } else {
    view_zenith_angle_radians_ += (previous_mouse_y_ - mouse_y) / kScale;
    view_zenith_angle_radians_ =
        std::max(0.0, std::min(kPi / 2.0, view_zenith_angle_radians_));
    view_azimuth_angle_radians_ += (previous_mouse_x_ - mouse_x) / kScale;
  }
  previous_mouse_x_ = mouse_x;
  previous_mouse_y_ = mouse_y;
}

void Demo::HandleMouseWheelEvent(int mouse_wheel_direction) {
  if (mouse_wheel_direction < 0) {
    view_distance_meters_ *= 1.05;
  } else {
    view_distance_meters_ /= 1.05;
  }
}

void Demo::SetView(double view_distance_meters,
    double view_zenith_angle_radians, double view_azimuth_angle_radians,
    double sun_zenith_angle_radians, double sun_azimuth_angle_radians,
    double exposure) {
  view_distance_meters_ = view_distance_meters;
  view_zenith_angle_radians_ = view_zenith_angle_radians;
  view_azimuth_angle_radians_ = view_azimuth_angle_radians;
  sun_zenith_angle_radians_ = sun_zenith_angle_radians;
  sun_azimuth_angle_radians_ = sun_azimuth_angle_radians;
  exposure_ = exposure;
}

}  // namespace demo
}  // namespace atmosphere
