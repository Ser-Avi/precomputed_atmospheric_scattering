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

/*<h2>atmosphere/demo/demo.glsl</h2>

<p>This GLSL fragment shader is used to render our demo scene, which consists of
a sphere S on a purely spherical planet P. It is rendered by "ray tracing", i.e.
the vertex shader outputs the view ray direction, and the fragment shader
computes the intersection of this ray with the spheres S and P to produce the
final pixels. The fragment shader also computes the intersection of the light
rays with the sphere S, to compute shadows, as well as the intersections of the
view ray with the shadow volume of S, in order to compute light shafts.

<p>Our fragment shader has the following inputs and outputs:
*/

uniform vec3 camera;
uniform float exposure;
uniform vec3 white_point;
uniform vec3 earth_center;
uniform vec3 sun_direction;
uniform vec2 sun_size;
in vec3 view_ray;
layout(location = 0) out vec4 color;


// avi added
uniform int display_texture;
uniform int scatter_slice;
uniform sampler2D transmittance_texture;
uniform sampler2D irradiance_texture;;
uniform sampler3D scattering_texture;

/*
<p>It uses the following constants, as well as the following atmosphere
rendering functions, defined externally (by the <code>Model</code>'s
<code>GetShader()</code> shader). The <code>USE_LUMINANCE</code> option is used
to select either the functions returning radiance values, or those returning
luminance values (see <a href="../model.h.html">model.h</a>).
*/

const float PI = 3.14159265;
const vec3 kSphereCenter = vec3(0.0, 0.0, 1000.0) / kLengthUnitInMeters;
const float kSphereRadius = 1000.0 / kLengthUnitInMeters;
const vec3 kSphereAlbedo = vec3(0.8);
const vec3 kGroundAlbedo = vec3(0.0, 0.0, 0.04);


vec3 GetSolarLuminance();
vec3 GetSkyLuminance(vec3 camera, vec3 view_ray, float shadow_length,
    vec3 sun_direction, out vec3 transmittance);
vec3 GetSkyLuminanceToPoint(vec3 camera, vec3 point, float shadow_length,
    vec3 sun_direction, out vec3 transmittance);
vec3 GetSunAndSkyIlluminance(
    vec3 p, vec3 normal, vec3 sun_direction, out vec3 sky_irradiance);

/*<h3>Shadows and light shafts</h3>

<p>The functions to compute shadows and light shafts must be defined before we
can use them in the main shader function, so we define them first. Testing if
a point is in the shadow of the sphere S is equivalent to test if the
corresponding light ray intersects the sphere, which is very simple to do.
However, this is only valid for a punctual light source, which is not the case
of the Sun. In the following function we compute an approximate (and biased)
soft shadow by taking the angular size of the Sun into account:
*/

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
      // Compute the distance between the view ray and the sphere, and the
      // corresponding (tangent of the) subtended angle. Finally, use this to
      // compute an approximate sun visibility.
      float ray_sphere_distance =
          kSphereRadius - sqrt(ray_sphere_center_squared_distance);
      float ray_sphere_angular_distance = -ray_sphere_distance / p_dot_v;
      return smoothstep(1.0, 0.0, ray_sphere_angular_distance / sun_size.x);
    }
  }
  return 1.0;
}

/*
<p>The sphere also partially occludes the sky light, and we approximate this
effect with an ambient occlusion factor. The ambient occlusion factor due to a
sphere is given in <a href=
"http://webserver.dmt.upm.es/~isidoro/tc3/Radiation%20View%20factors.pdf"
>Radiation View Factors</a> (Isidoro Martinez, 1995). In the simple case where
the sphere is fully visible, it is given by the following function:
*/

float GetSkyVisibility(vec3 point) {
  vec3 p = point - kSphereCenter;
  float p_dot_p = dot(p, p);
  return
      1.0 + p.z / sqrt(p_dot_p) * kSphereRadius * kSphereRadius / p_dot_p;
}

/*
<p>To compute light shafts we need the intersections of the view ray with the
shadow volume of the sphere S. Since the Sun is not a punctual light source this
shadow volume is not a cylinder but a cone (for the umbra, plus another cone for
the penumbra, but we ignore it here):

<svg width="505px" height="200px">
  <style type="text/css"><![CDATA[
    circle { fill: #000000; stroke: none; }
    path { fill: none; stroke: #000000; }
    text { font-size: 16px; font-style: normal; font-family: Sans; }
    .vector { font-weight: bold; }
  ]]></style>
  <path d="m 10,75 455,120"/>
  <path d="m 10,125 455,-120"/>
  <path d="m 120,50 160,130"/>
  <path d="m 138,70 7,0 0,-7"/>
  <path d="m 410,65 40,0 m -5,-5 5,5 -5,5"/>
  <path d="m 20,100 430,0" style="stroke-dasharray:8,4,2,4;"/>
  <path d="m 255,25 0,155" style="stroke-dasharray:2,2;"/>
  <path d="m 280,160 -25,0" style="stroke-dasharray:2,2;"/>
  <path d="m 255,140 60,0" style="stroke-dasharray:2,2;"/>
  <path d="m 300,105 5,-5 5,5 m -5,-5 0,40 m -5,-5 5,5 5,-5"/>
  <path d="m 265,105 5,-5 5,5 m -5,-5 0,60 m -5,-5 5,5 5,-5"/>
  <path d="m 260,80 -5,5 5,5 m -5,-5 85,0 m -5,5 5,-5 -5,-5"/>
  <path d="m 335,95 5,5 5,-5 m -5,5 0,-60 m -5,5 5,-5 5,5"/>
  <path d="m 50,100 a 50,50 0 0 1 2,-14" style="stroke-dasharray:2,1;"/>
  <circle cx="340" cy="100" r="60" style="fill: none; stroke: #000000;"/>
  <circle cx="340" cy="100" r="2.5"/>
  <circle cx="255" cy="160" r="2.5"/>
  <circle cx="120" cy="50" r="2.5"/>
  <text x="105" y="45" class="vector">p</text>
  <text x="240" y="170" class="vector">q</text>
  <text x="425" y="55" class="vector">s</text>
  <text x="135" y="55" class="vector">v</text>
  <text x="345" y="75">R</text>
  <text x="275" y="135">r</text>
  <text x="310" y="125">ρ</text>
  <text x="215" y="120">d</text>
  <text x="290" y="80">δ</text>
  <text x="30" y="95">α</text>
</svg>

<p>Noting, as in the above figure, $\bp$ the camera position, $\bv$ and $\bs$
the unit view ray and sun direction vectors and $R$ the sphere radius (supposed
to be centered on the origin), the point at distance $d$ from the camera is
$\bq=\bp+d\bv$. This point is at a distance $\delta=-\bq\cdot\bs$ from the
sphere center along the umbra cone axis, and at a distance $r$ from this axis
given by $r^2=\bq\cdot\bq-\delta^2$. Finally, at distance $\delta$ along the
axis the umbra cone has radius $\rho=R-\delta\tan\alpha$, where $\alpha$ is
the Sun's angular radius. The point at distance $d$ from the camera is on the
shadow cone only if $r^2=\rho^2$, i.e. only if
\begin{equation}
(\bp+d\bv)\cdot(\bp+d\bv)-((\bp+d\bv)\cdot\bs)^2=
(R+((\bp+d\bv)\cdot\bs)\tan\alpha)^2
\end{equation}
Developping this gives a quadratic equation for $d$:
\begin{equation}
ad^2+2bd+c=0
\end{equation}
where
<ul>
<li>$a=1-l(\bv\cdot\bs)^2$,</li>
<li>$b=\bp\cdot\bv-l(\bp\cdot\bs)(\bv\cdot\bs)-\tan(\alpha)R(\bv\cdot\bs)$,</li>
<li>$c=\bp\cdot\bp-l(\bp\cdot\bs)^2-2\tan(\alpha)R(\bp\cdot\bs)-R^2$,</li>
<li>$l=1+\tan^2\alpha$</li>
</ul>
From this we deduce the two possible solutions for $d$, which must be clamped to
the actual shadow part of the mathematical cone (i.e. the slab between the
sphere center and the cone apex or, in other words, the points for which
$\delta$ is between $0$ and $R/\tan\alpha$). The following function implements
these equations:
*/

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
    // The values of d for which delta is equal to 0 and kSphereRadius / k.
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

/*<h3>Main shading function</h3>

<p>Using these functions we can now implement the main shader function, which
computes the radiance from the scene for a given view ray. This function first
tests if the view ray intersects the sphere S. If so it computes the sun and
sky light received by the sphere at the intersection point, combines this with
the sphere BRDF and the aerial perspective between the camera and the sphere.
It then does the same with the ground, i.e. with the planet sphere P, and then
computes the sky radiance and transmittance. Finally, all these terms are
composited together (an opacity is also computed for each object, using an
approximate view cone - sphere intersection factor) to get the final radiance.

<p>We start with the computation of the intersections of the view ray with the
shadow volume of the sphere, because they are needed to get the aerial
perspective for the sphere and the planet:
*/

void main() {
    if (display_texture == 0)
    {
        // Normalized view direction vector.
        vec3 view_direction = normalize(view_ray);
        // Tangent of the angle subtended by this fragment.
        float fragment_angular_size =
            length(dFdx(view_ray) + dFdy(view_ray)) / length(view_ray);

        float shadow_in;
        float shadow_out;
        GetSphereShadowInOut(view_direction, sun_direction, shadow_in, shadow_out);

        // Hack to fade out light shafts when the Sun is very close to the horizon.
        float lightshaft_fadein_hack = smoothstep(
            0.02, 0.04, dot(normalize(camera - earth_center), sun_direction));

        /*
        <p>We then test whether the view ray intersects the sphere S or not. If it does,
        we compute an approximate (and biased) opacity value, using the same
        approximation as in <code>GetSunVisibility</code>:
        */

        // Compute the distance between the view ray line and the sphere center,
        // and the distance between the camera and the intersection of the view
        // ray with the sphere (or NaN if there is no intersection).
        vec3 p = camera - kSphereCenter;
        float p_dot_v = dot(p, view_direction);
        float p_dot_p = dot(p, p);
        float ray_sphere_center_squared_distance = p_dot_p - p_dot_v * p_dot_v;
        float discriminant =
            kSphereRadius * kSphereRadius - ray_sphere_center_squared_distance;

        // Compute the radiance reflected by the sphere, if the ray intersects it.
        float sphere_alpha = 0.0;
        vec3 sphere_radiance = vec3(0.0);
        if (discriminant >= 0.0) {
            float distance_to_intersection = -p_dot_v - sqrt(discriminant);
            if (distance_to_intersection > 0.0) {
                // Compute the distance between the view ray and the sphere, and the
                // corresponding (tangent of the) subtended angle. Finally, use this to
                // compute the approximate analytic antialiasing factor sphere_alpha.
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

                // Compute the radiance reflected by the sphere.
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

        // Compute the distance between the view ray line and the Earth center,
        // and the distance between the camera and the intersection of the view
        // ray with the ground (or NaN if there is no intersection).
        p = camera - earth_center;
        p_dot_v = dot(p, view_direction);
        p_dot_p = dot(p, p);
        float ray_earth_center_squared_distance = p_dot_p - p_dot_v * p_dot_v;
        discriminant =
            earth_center.z * earth_center.z - ray_earth_center_squared_distance;

        // Compute the radiance reflected by the ground, if the ray intersects it.
        float ground_alpha = 0.0;
        vec3 ground_radiance = vec3(0.0);
        if (discriminant >= 0.0) {
            float distance_to_intersection = -p_dot_v - sqrt(discriminant);
            if (distance_to_intersection > 0.0) {
                vec3 point = camera + view_direction * distance_to_intersection;
                vec3 normal = normalize(point - earth_center);

                // Compute the radiance reflected by the ground.
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

        // Compute the radiance of the sky.
        float shadow_length = max(0.0, shadow_out - shadow_in) *
            lightshaft_fadein_hack;
        vec3 transmittance;
        vec3 radiance = GetSkyLuminance(
            camera - earth_center, view_direction, shadow_length, sun_direction,
            transmittance);

        // If the view ray intersects the Sun, add the Sun radiance.
        if (dot(view_direction, sun_direction) > sun_size.y) {
            radiance = radiance + transmittance * GetSolarLuminance();
        }
        radiance = mix(radiance, ground_radiance, ground_alpha);
        radiance = mix(radiance, sphere_radiance, sphere_alpha);
        color.rgb =
            pow(vec3(1.0) - exp(-radiance / white_point * exposure), vec3(1.0 / 2.2));
        color.a = 1.0;
    }
    // Display transmittance texture
    else if (display_texture == 1) {
        vec2 texCoord = gl_FragCoord.xy / vec2(256.f, 64.f);
        color = texture(transmittance_texture, texCoord);
    }
    // Display irradiance texture
    else if (display_texture == 2) {
        vec2 texCoord = gl_FragCoord.xy / vec2(64.f, 16.f);
        color = texture(irradiance_texture, texCoord);
    }
    // display scattering texture slice
    else if (display_texture == 3) {
        vec3 texCoord = vec3(vec2(gl_FragCoord.xy / vec2(8.f * 32.f, 128.f)), scatter_slice * 1.f);
        color = texture(scattering_texture, texCoord);
    }
}


//const AtmosphereParameters ATMOSPHERE = AtmosphereParameters(    vec3(1.474000, 1.850400, 1.911980),
//    0.004675,
//    6360.000000,
//    6420.000000,
//    DensityProfile(DensityProfileLayer[2](DensityProfileLayer(0.000000, 0.000000, 0.000000, 0.000000, 0.000000), DensityProfileLayer(0.000000, 1.000000, -0.125000, 0.000000, 0.000000))),
//    vec3(0.005802, 0.013558, 0.033100),
//    DensityProfile(DensityProfileLayer[2](DensityProfileLayer(0.000000, 0.000000, 0.000000, 0.000000, 0.000000), DensityProfileLayer(0.000000, 1.000000, -0.833333, 0.000000, 0.000000))),
//    vec3(0.003996, 0.003996, 0.003996),
//    vec3(0.004440, 0.004440, 0.004440),
//    0.800000,
//    DensityProfile(DensityProfileLayer[2](DensityProfileLayer(25.000000, 0.000000, 0.000000, 0.066667, -0.666667), DensityProfileLayer(0.000000, 0.000000, 0.000000, -0.066667, 2.666667))),
//    vec3(0.000650, 0.001881, 0.000085),
//    vec3(0.100000, 0.100000, 0.100000),
//    -0.500000);
//
//const vec3 SKY_SPECTRAL_RADIANCE_TO_LUMINANCE = vec3(683.000000, 683.000000, 683.000000);
//const vec3 SUN_SPECTRAL_RADIANCE_TO_LUMINANCE = vec3(98242.786222, 69954.398112, 66475.012354);
//
//const float kLengthUnitInMeters = 1000.000000;
//uniform vec3 camera;
//uniform float exposure;
//uniform vec3 white_point;
//uniform vec3 earth_center;
//uniform vec3 sun_direction;
//uniform vec2 sun_size;
//in vec3 view_ray;
//layout(location = 0) out vec4 color;
//uniform int display_texture;
//uniform int scatter_slice;
//uniform sampler2D transmittance_texture;
//uniform sampler2D irradiance_texture;
//uniform sampler3D scattering_texture;
//
//const vec3 kSphereCenter = vec3(0.0, 0.0, 1000.0) / kLengthUnitInMeters;
//const float kSphereRadius = 1000.0 / kLengthUnitInMeters;
//const vec3 kSphereAlbedo = vec3(0.8);
//const vec3 kGroundAlbedo = vec3(0.0, 0.0, 0.04);
//
//// Utility functions from full_shader.glsl
//Number ClampCosine(Number mu) {
//    return clamp(mu, Number(-1.0), Number(1.0));
//}
//
//Length ClampDistance(Length d) {
//    return max(d, 0.0 * m);
//}
//
//Length ClampRadius(IN(AtmosphereParameters) atmosphere, Length r) {
//    return clamp(r, atmosphere.bottom_radius, atmosphere.top_radius);
//}
//
//Length SafeSqrt(Area a) {
//    return sqrt(max(a, 0.0 * m2));
//}
//
//Length DistanceToTopAtmosphereBoundary(IN(AtmosphereParameters) atmosphere,
//    Length r, Number mu) {
//    assert(r <= atmosphere.top_radius);
//    assert(mu >= -1.0 && mu <= 1.0);
//    Area discriminant = r * r * (mu * mu - 1.0) +
//        atmosphere.top_radius * atmosphere.top_radius;
//    return ClampDistance(-r * mu + SafeSqrt(discriminant));
//}
//
//bool RayIntersectsGround(IN(AtmosphereParameters) atmosphere,
//    Length r, Number mu) {
//    assert(r >= atmosphere.bottom_radius);
//    assert(mu >= -1.0 && mu <= 1.0);
//    return mu < 0.0 && r * r * (mu * mu - 1.0) +
//        atmosphere.bottom_radius * atmosphere.bottom_radius >= 0.0 * m2;
//}
//
//Number GetLayerDensity(IN(DensityProfileLayer) layer, Length altitude) {
//    Number density = layer.exp_term * exp(layer.exp_scale * altitude) +
//        layer.linear_term * altitude + layer.constant_term;
//    return clamp(density, Number(0.0), Number(1.0));
//}
//
//Number GetProfileDensity(IN(DensityProfile) profile, Length altitude) {
//    return altitude < profile.layers[0].width ?
//        GetLayerDensity(profile.layers[0], altitude) :
//        GetLayerDensity(profile.layers[1], altitude);
//}
//
//Number GetTextureCoordFromUnitRange(Number x, int texture_size) {
//    return 0.5 / Number(texture_size) + x * (1.0 - 1.0 / Number(texture_size));
//}
//
//vec2 GetTransmittanceTextureUvFromRMu(IN(AtmosphereParameters) atmosphere,
//    Length r, Number mu) {
//    assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
//    assert(mu >= -1.0 && mu <= 1.0);
//    Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
//        atmosphere.bottom_radius * atmosphere.bottom_radius);
//    Length rho =
//        SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
//    Length d = DistanceToTopAtmosphereBoundary(atmosphere, r, mu);
//    Length d_min = atmosphere.top_radius - r;
//    Length d_max = rho + H;
//    Number x_mu = (d - d_min) / (d_max - d_min);
//    Number x_r = rho / H;
//    return vec2(GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_TEXTURE_WIDTH),
//        GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_TEXTURE_HEIGHT));
//}
//
//DimensionlessSpectrum GetTransmittanceToTopAtmosphereBoundary(
//    IN(AtmosphereParameters) atmosphere,
//    IN(TransmittanceTexture) transmittance_texture,
//    Length r, Number mu) {
//    assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
//    vec2 uv = GetTransmittanceTextureUvFromRMu(atmosphere, r, mu);
//    return DimensionlessSpectrum(texture(transmittance_texture, uv));
//}
//
//DimensionlessSpectrum GetTransmittance(
//    IN(AtmosphereParameters) atmosphere,
//    IN(TransmittanceTexture) transmittance_texture,
//    Length r, Number mu, Length d, bool ray_r_mu_intersects_ground) {
//    assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
//    assert(mu >= -1.0 && mu <= 1.0);
//    assert(d >= 0.0 * m);
//    Length r_d = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
//    Number mu_d = ClampCosine((r * mu + d) / r_d);
//    if (ray_r_mu_intersects_ground) {
//        return min(
//            GetTransmittanceToTopAtmosphereBoundary(
//                atmosphere, transmittance_texture, r_d, -mu_d) /
//            GetTransmittanceToTopAtmosphereBoundary(
//                atmosphere, transmittance_texture, r, -mu),
//            DimensionlessSpectrum(1.0));
//    }
//    else {
//        return min(
//            GetTransmittanceToTopAtmosphereBoundary(
//                atmosphere, transmittance_texture, r, mu) /
//            GetTransmittanceToTopAtmosphereBoundary(
//                atmosphere, transmittance_texture, r_d, mu_d),
//            DimensionlessSpectrum(1.0));
//    }
//}
//
//DimensionlessSpectrum GetTransmittanceToSun(
//    IN(AtmosphereParameters) atmosphere,
//    IN(TransmittanceTexture) transmittance_texture,
//    Length r, Number mu_s) {
//    Number sin_theta_h = atmosphere.bottom_radius / r;
//    Number cos_theta_h = -sqrt(max(1.0 - sin_theta_h * sin_theta_h, 0.0));
//    return GetTransmittanceToTopAtmosphereBoundary(
//        atmosphere, transmittance_texture, r, mu_s) *
//        smoothstep(-sin_theta_h * atmosphere.sun_angular_radius / rad,
//            sin_theta_h * atmosphere.sun_angular_radius / rad,
//            mu_s - cos_theta_h);
//}
//
//InverseSolidAngle RayleighPhaseFunction(Number nu) {
//    InverseSolidAngle k = 3.0 / (16.0 * PI * sr);
//    return k * (1.0 + nu * nu);
//}
//
//InverseSolidAngle MiePhaseFunction(Number g, Number nu) {
//    InverseSolidAngle k = 3.0 / (8.0 * PI * sr) * (1.0 - g * g) / (2.0 + g * g);
//    return k * (1.0 + nu * nu) / pow(1.0 + g * g - 2.0 * g * nu, 1.5);
//}
//
//vec4 GetScatteringTextureUvwzFromRMuMuSNu(IN(AtmosphereParameters) atmosphere,
//    Length r, Number mu, Number mu_s, Number nu,
//    bool ray_r_mu_intersects_ground) {
//    assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
//    assert(mu >= -1.0 && mu <= 1.0);
//    assert(mu_s >= -1.0 && mu_s <= 1.0);
//    assert(nu >= -1.0 && nu <= 1.0);
//    Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
//        atmosphere.bottom_radius * atmosphere.bottom_radius);
//    Length rho =
//        SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
//    Number u_r = GetTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE);
//    Length r_mu = r * mu;
//    Area discriminant =
//        r_mu * r_mu - r * r + atmosphere.bottom_radius * atmosphere.bottom_radius;
//    Number u_mu;
//    if (ray_r_mu_intersects_ground) {
//        Length d = -r_mu - SafeSqrt(discriminant);
//        Length d_min = r - atmosphere.bottom_radius;
//        Length d_max = rho;
//        u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(d_max == d_min ? 0.0 :
//            (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
//    }
//    else {
//        Length d = -r_mu + SafeSqrt(discriminant + H * H);
//        Length d_min = atmosphere.top_radius - r;
//        Length d_max = rho + H;
//        u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
//            (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
//    }
//    Length d = DistanceToTopAtmosphereBoundary(
//        atmosphere, atmosphere.bottom_radius, mu_s);
//    Length d_min = atmosphere.top_radius - atmosphere.bottom_radius;
//    Length d_max = H;
//    Number a = (d - d_min) / (d_max - d_min);
//    Length D = DistanceToTopAtmosphereBoundary(
//        atmosphere, atmosphere.bottom_radius, atmosphere.mu_s_min);
//    Number A = (D - d_min) / (d_max - d_min);
//    Number u_mu_s = GetTextureCoordFromUnitRange(
//        max(1.0 - a / A, 0.0) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE);
//    Number u_nu = (nu + 1.0) / 2.0;
//    return vec4(u_nu, u_mu_s, u_mu, u_r);
//}
//
//vec3 GetExtrapolatedSingleMieScattering(
//    IN(AtmosphereParameters) atmosphere, IN(vec4) scattering) {
//    if (scattering.r <= 0.0) {
//        return vec3(0.0);
//    }
//    return scattering.rgb * scattering.a / scattering.r *
//        (atmosphere.rayleigh_scattering.r / atmosphere.mie_scattering.r) *
//        (atmosphere.mie_scattering / atmosphere.rayleigh_scattering);
//}
//
//IrradianceSpectrum GetCombinedScattering(
//    IN(AtmosphereParameters) atmosphere,
//    IN(ReducedScatteringTexture) scattering_texture,
//    IN(ReducedScatteringTexture) single_mie_scattering_texture,
//    Length r, Number mu, Number mu_s, Number nu,
//    bool ray_r_mu_intersects_ground,
//    OUT(IrradianceSpectrum) single_mie_scattering) {
//    vec4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(
//        atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
//    Number tex_coord_x = uvwz.x * Number(SCATTERING_TEXTURE_NU_SIZE - 1);
//    Number tex_x = floor(tex_coord_x);
//    Number lerp = tex_coord_x - tex_x;
//    vec3 uvw0 = vec3((tex_x + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),
//        uvwz.z, uvwz.w);
//    vec3 uvw1 = vec3((tex_x + 1.0 + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),
//        uvwz.z, uvwz.w);
//    vec4 combined_scattering =
//        texture(scattering_texture, uvw0) * (1.0 - lerp) +
//        texture(scattering_texture, uvw1) * lerp;
//    IrradianceSpectrum scattering = IrradianceSpectrum(combined_scattering);
//    single_mie_scattering =
//        GetExtrapolatedSingleMieScattering(atmosphere, combined_scattering);
//    return scattering;
//}
//
//RadianceSpectrum GetSkyRadiance(
//    IN(AtmosphereParameters) atmosphere,
//    IN(TransmittanceTexture) transmittance_texture,
//    IN(ReducedScatteringTexture) scattering_texture,
//    IN(ReducedScatteringTexture) single_mie_scattering_texture,
//    Position camera, IN(Direction) view_ray, Length shadow_length,
//    IN(Direction) sun_direction, OUT(DimensionlessSpectrum) transmittance) {
//    Length r = length(camera);
//    Length rmu = dot(camera, view_ray);
//    Length distance_to_top_atmosphere_boundary = -rmu -
//        sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);
//    if (distance_to_top_atmosphere_boundary > 0.0 * m) {
//        camera = camera + view_ray * distance_to_top_atmosphere_boundary;
//        r = atmosphere.top_radius;
//        rmu += distance_to_top_atmosphere_boundary;
//    }
//    else if (r > atmosphere.top_radius) {
//        transmittance = DimensionlessSpectrum(1.0);
//        return RadianceSpectrum(0.0 * watt_per_square_meter_per_sr_per_nm);
//    }
//    Number mu = rmu / r;
//    Number mu_s = dot(camera, sun_direction) / r;
//    Number nu = dot(view_ray, sun_direction);
//    bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);
//    transmittance = ray_r_mu_intersects_ground ? DimensionlessSpectrum(0.0) :
//        GetTransmittanceToTopAtmosphereBoundary(
//            atmosphere, transmittance_texture, r, mu);
//    IrradianceSpectrum single_mie_scattering;
//    IrradianceSpectrum scattering;
//    if (shadow_length == 0.0 * m) {
//        scattering = GetCombinedScattering(
//            atmosphere, scattering_texture, single_mie_scattering_texture,
//            r, mu, mu_s, nu, ray_r_mu_intersects_ground,
//            single_mie_scattering);
//    }
//    else {
//        Length d = shadow_length;
//        Length r_p =
//            ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
//        Number mu_p = (r * mu + d) / r_p;
//        Number mu_s_p = (r * mu_s + d * nu) / r_p;
//        scattering = GetCombinedScattering(
//            atmosphere, scattering_texture, single_mie_scattering_texture,
//            r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,
//            single_mie_scattering);
//        DimensionlessSpectrum shadow_transmittance =
//            GetTransmittance(atmosphere, transmittance_texture,
//                r, mu, shadow_length, ray_r_mu_intersects_ground);
//        scattering = scattering * shadow_transmittance;
//        single_mie_scattering = single_mie_scattering * shadow_transmittance;
//    }
//    return scattering * RayleighPhaseFunction(nu) + single_mie_scattering *
//        MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
//}
//
//RadianceSpectrum GetSkyRadianceToPoint(
//    IN(AtmosphereParameters) atmosphere,
//    IN(TransmittanceTexture) transmittance_texture,
//    IN(ReducedScatteringTexture) scattering_texture,
//    IN(ReducedScatteringTexture) single_mie_scattering_texture,
//    Position camera, IN(Position) point, Length shadow_length,
//    IN(Direction) sun_direction, OUT(DimensionlessSpectrum) transmittance) {
//    Direction view_ray = normalize(point - camera);
//    Length r = length(camera);
//    Length rmu = dot(camera, view_ray);
//    Length distance_to_top_atmosphere_boundary = -rmu -
//        sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);
//    if (distance_to_top_atmosphere_boundary > 0.0 * m) {
//        camera = camera + view_ray * distance_to_top_atmosphere_boundary;
//        r = atmosphere.top_radius;
//        rmu += distance_to_top_atmosphere_boundary;
//    }
//    Number mu = rmu / r;
//    Number mu_s = dot(camera, sun_direction) / r;
//    Number nu = dot(view_ray, sun_direction);
//    Length d = length(point - camera);
//    bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);
//    transmittance = GetTransmittance(atmosphere, transmittance_texture,
//        r, mu, d, ray_r_mu_intersects_ground);
//    IrradianceSpectrum single_mie_scattering;
//    IrradianceSpectrum scattering = GetCombinedScattering(
//        atmosphere, scattering_texture, single_mie_scattering_texture,
//        r, mu, mu_s, nu, ray_r_mu_intersects_ground,
//        single_mie_scattering);
//    d = max(d - shadow_length, 0.0 * m);
//    Length r_p = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
//    Number mu_p = (r * mu + d) / r_p;
//    Number mu_s_p = (r * mu_s + d * nu) / r_p;
//    IrradianceSpectrum single_mie_scattering_p;
//    IrradianceSpectrum scattering_p = GetCombinedScattering(
//        atmosphere, scattering_texture, single_mie_scattering_texture,
//        r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,
//        single_mie_scattering_p);
//    DimensionlessSpectrum shadow_transmittance = transmittance;
//    if (shadow_length > 0.0 * m) {
//        shadow_transmittance = GetTransmittance(atmosphere, transmittance_texture,
//            r, mu, d, ray_r_mu_intersects_ground);
//    }
//    scattering = scattering - shadow_transmittance * scattering_p;
//    single_mie_scattering =
//        single_mie_scattering - shadow_transmittance * single_mie_scattering_p;
//    single_mie_scattering = GetExtrapolatedSingleMieScattering(
//        atmosphere, vec4(scattering, single_mie_scattering.r));
//    single_mie_scattering = single_mie_scattering *
//        smoothstep(Number(0.0), Number(0.01), mu_s);
//    return scattering * RayleighPhaseFunction(nu) + single_mie_scattering *
//        MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
//}
//
//vec2 GetIrradianceTextureUvFromRMuS(IN(AtmosphereParameters) atmosphere,
//    Length r, Number mu_s) {
//    assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
//    assert(mu_s >= -1.0 && mu_s <= 1.0);
//    Number x_r = (r - atmosphere.bottom_radius) /
//        (atmosphere.top_radius - atmosphere.bottom_radius);
//    Number x_mu_s = mu_s * 0.5 + 0.5;
//    return vec2(GetTextureCoordFromUnitRange(x_mu_s, IRRADIANCE_TEXTURE_WIDTH),
//        GetTextureCoordFromUnitRange(x_r, IRRADIANCE_TEXTURE_HEIGHT));
//}
//
//IrradianceSpectrum GetIrradiance(
//    IN(AtmosphereParameters) atmosphere,
//    IN(IrradianceTexture) irradiance_texture,
//    Length r, Number mu_s) {
//    vec2 uv = GetIrradianceTextureUvFromRMuS(atmosphere, r, mu_s);
//    return IrradianceSpectrum(texture(irradiance_texture, uv));
//}
//
//IrradianceSpectrum GetSunAndSkyIrradiance(
//    IN(AtmosphereParameters) atmosphere,
//    IN(TransmittanceTexture) transmittance_texture,
//    IN(IrradianceTexture) irradiance_texture,
//    IN(Position) point, IN(Direction) normal, IN(Direction) sun_direction,
//    OUT(IrradianceSpectrum) sky_irradiance) {
//    Length r = length(point);
//    Number mu_s = dot(point, sun_direction) / r;
//    sky_irradiance = GetIrradiance(atmosphere, irradiance_texture, r, mu_s) *
//        (1.0 + dot(normal, point) / r) * 0.5;
//    return atmosphere.solar_irradiance *
//        GetTransmittanceToSun(
//            atmosphere, transmittance_texture, r, mu_s) *
//        max(dot(normal, sun_direction), 0.0);
//}
//
//// Original demo wrapper functions - unchanged
//vec3 GetSolarLuminance();
//vec3 GetSkyLuminance(vec3 camera, vec3 view_ray, float shadow_length,
//    vec3 sun_direction, out vec3 transmittance);
//vec3 GetSkyLuminanceToPoint(vec3 camera, vec3 point, float shadow_length,
//    vec3 sun_direction, out vec3 transmittance);
//vec3 GetSunAndSkyIlluminance(
//    vec3 p, vec3 normal, vec3 sun_direction, out vec3 sky_irradiance);
//
//vec3 GetSolarLuminance() {
//    return ATMOSPHERE.solar_irradiance /
//        (PI * ATMOSPHERE.sun_angular_radius * ATMOSPHERE.sun_angular_radius);
//}
//
//vec3 GetSkyLuminance(
//    vec3 camera, vec3 view_ray, float shadow_length,
//    vec3 sun_direction, out vec3 transmittance) {
//    return GetSkyRadiance(ATMOSPHERE, transmittance_texture,
//        scattering_texture, scattering_texture,
//        camera, view_ray, shadow_length, sun_direction, transmittance);
//}
//
//vec3 GetSkyLuminanceToPoint(
//    vec3 camera, vec3 point, float shadow_length,
//    vec3 sun_direction, out vec3 transmittance) {
//    return GetSkyRadianceToPoint(ATMOSPHERE, transmittance_texture,
//        scattering_texture, scattering_texture,
//        camera, point, shadow_length, sun_direction, transmittance);
//}
//
//vec3 GetSunAndSkyIlluminance(
//    vec3 p, vec3 normal, vec3 sun_direction, out vec3 sky_irradiance) {
//    return GetSunAndSkyIrradiance(
//        ATMOSPHERE, transmittance_texture, irradiance_texture,
//        p, normal, sun_direction, sky_irradiance);
//}
//
//float GetSunVisibility(vec3 point, vec3 sun_direction) {
//    vec3 p = point - kSphereCenter;
//    float p_dot_v = dot(p, sun_direction);
//    float p_dot_p = dot(p, p);
//    float ray_sphere_center_squared_distance = p_dot_p - p_dot_v * p_dot_v;
//    float discriminant =
//        kSphereRadius * kSphereRadius - ray_sphere_center_squared_distance;
//    if (discriminant >= 0.0) {
//        float distance_to_intersection = -p_dot_v - sqrt(discriminant);
//        if (distance_to_intersection > 0.0) {
//            float ray_sphere_distance =
//                kSphereRadius - sqrt(ray_sphere_center_squared_distance);
//            float ray_sphere_angular_distance = -ray_sphere_distance / p_dot_v;
//            return smoothstep(1.0, 0.0, ray_sphere_angular_distance / sun_size.x);
//        }
//    }
//    return 1.0;
//}
//
//float GetSkyVisibility(vec3 point) {
//    vec3 p = point - kSphereCenter;
//    float p_dot_p = dot(p, p);
//    return
//        1.0 + p.z / sqrt(p_dot_p) * kSphereRadius * kSphereRadius / p_dot_p;
//}
//
//void GetSphereShadowInOut(vec3 view_direction, vec3 sun_direction,
//    out float d_in, out float d_out) {
//    vec3 pos = camera - kSphereCenter;
//    float pos_dot_sun = dot(pos, sun_direction);
//    float view_dot_sun = dot(view_direction, sun_direction);
//    float k = sun_size.x;
//    float l = 1.0 + k * k;
//    float a = 1.0 - l * view_dot_sun * view_dot_sun;
//    float b = dot(pos, view_direction) - l * pos_dot_sun * view_dot_sun -
//        k * kSphereRadius * view_dot_sun;
//    float c = dot(pos, pos) - l * pos_dot_sun * pos_dot_sun -
//        2.0 * k * kSphereRadius * pos_dot_sun - kSphereRadius * kSphereRadius;
//    float discriminant = b * b - a * c;
//    if (discriminant > 0.0) {
//        d_in = max(0.0, (-b - sqrt(discriminant)) / a);
//        d_out = (-b + sqrt(discriminant)) / a;
//        float d_base = -pos_dot_sun / view_dot_sun;
//        float d_apex = -(pos_dot_sun + kSphereRadius / k) / view_dot_sun;
//        if (view_dot_sun > 0.0) {
//            d_in = max(d_in, d_apex);
//            d_out = a > 0.0 ? min(d_out, d_base) : d_base;
//        }
//        else {
//            d_in = a > 0.0 ? max(d_in, d_base) : d_base;
//            d_out = min(d_out, d_apex);
//        }
//    }
//    else {
//        d_in = 0.0;
//        d_out = 0.0;
//    }
//}
//
//void main() {
//    if (display_texture == 0)
//    {
//        vec3 view_direction = normalize(view_ray);
//        float fragment_angular_size =
//            length(dFdx(view_ray) + dFdy(view_ray)) / length(view_ray);
//        float shadow_in;
//        float shadow_out;
//        GetSphereShadowInOut(view_direction, sun_direction, shadow_in, shadow_out);
//        float lightshaft_fadein_hack = smoothstep(
//            0.02, 0.04, dot(normalize(camera - earth_center), sun_direction));
//        /*
//        <p>We then test whether the view ray intersects the sphere S or not. If it does,
//        we compute an approximate (and biased) opacity value, using the same
//        approximation as in <code>GetSunVisibility</code>:
//        */
//        vec3 p = camera - kSphereCenter;
//        float p_dot_v = dot(p, view_direction);
//        float p_dot_p = dot(p, p);
//        float ray_sphere_center_squared_distance = p_dot_p - p_dot_v * p_dot_v;
//        float discriminant =
//            kSphereRadius * kSphereRadius - ray_sphere_center_squared_distance;
//        float sphere_alpha = 0.0;
//        vec3 sphere_radiance = vec3(0.0);
//        if (discriminant >= 0.0) {
//            float distance_to_intersection = -p_dot_v - sqrt(discriminant);
//            if (distance_to_intersection > 0.0) {
//                float ray_sphere_distance =
//                    kSphereRadius - sqrt(ray_sphere_center_squared_distance);
//                float ray_sphere_angular_distance = -ray_sphere_distance / p_dot_v;
//                sphere_alpha =
//                    min(ray_sphere_angular_distance / fragment_angular_size, 1.0);
//                /*
//                <p>We can then compute the intersection point and its normal, and use them to
//                get the sun and sky irradiance received at this point. The reflected radiance
//                follows, by multiplying the irradiance with the sphere BRDF:
//                */
//                vec3 point = camera + view_direction * distance_to_intersection;
//                vec3 normal = normalize(point - kSphereCenter);
//                vec3 sky_irradiance;
//                vec3 sun_irradiance = GetSunAndSkyIlluminance(
//                    point - earth_center, normal, sun_direction, sky_irradiance);
//                sphere_radiance =
//                    kSphereAlbedo * (1.0 / PI) * (sun_irradiance + sky_irradiance);
//                /*
//                <p>Finally, we take into account the aerial perspective between the camera and
//                the sphere, which depends on the length of this segment which is in shadow:
//                */
//                float shadow_length =
//                    max(0.0, min(shadow_out, distance_to_intersection) - shadow_in) *
//                    lightshaft_fadein_hack;
//                vec3 transmittance;
//                vec3 in_scatter = GetSkyLuminanceToPoint(camera - earth_center,
//                    point - earth_center, shadow_length, sun_direction, transmittance);
//                sphere_radiance = sphere_radiance * transmittance + in_scatter;
//            }
//        }
//        /*
//        <p>In the following we repeat the same steps as above, but for the planet sphere
//        P instead of the sphere S (a smooth opacity is not really needed here, so we
//        don't compute it. Note also how we modulate the sun and sky irradiance received
//        on the ground by the sun and sky visibility factors):
//        */
//        p = camera - earth_center;
//        p_dot_v = dot(p, view_direction);
//        p_dot_p = dot(p, p);
//        float ray_earth_center_squared_distance = p_dot_p - p_dot_v * p_dot_v;
//        discriminant =
//            earth_center.z * earth_center.z - ray_earth_center_squared_distance;
//        float ground_alpha = 0.0;
//        vec3 ground_radiance = vec3(0.0);
//        if (discriminant >= 0.0) {
//            float distance_to_intersection = -p_dot_v - sqrt(discriminant);
//            if (distance_to_intersection > 0.0) {
//                vec3 point = camera + view_direction * distance_to_intersection;
//                vec3 normal = normalize(point - earth_center);
//                vec3 sky_irradiance;
//                vec3 sun_irradiance = GetSunAndSkyIlluminance(
//                    point - earth_center, normal, sun_direction, sky_irradiance);
//                ground_radiance = kGroundAlbedo * (1.0 / PI) * (
//                    sun_irradiance * GetSunVisibility(point, sun_direction) +
//                    sky_irradiance * GetSkyVisibility(point));
//                float shadow_length =
//                    max(0.0, min(shadow_out, distance_to_intersection) - shadow_in) *
//                    lightshaft_fadein_hack;
//                vec3 transmittance;
//                vec3 in_scatter = GetSkyLuminanceToPoint(camera - earth_center,
//                    point - earth_center, shadow_length, sun_direction, transmittance);
//                ground_radiance = ground_radiance * transmittance + in_scatter;
//                ground_alpha = 1.0;
//            }
//        }
//        /*
//        <p>Finally, we compute the radiance and transmittance of the sky, and composite
//        together, from back to front, the radiance and opacities of all the objects of
//        the scene:
//        */
//        float shadow_length = max(0.0, shadow_out - shadow_in) *
//            lightshaft_fadein_hack;
//        vec3 transmittance;
//        vec3 radiance = GetSkyLuminance(
//            camera - earth_center, view_direction, shadow_length, sun_direction,
//            transmittance);
//        if (dot(view_direction, sun_direction) > sun_size.y) {
//            radiance = radiance + transmittance * GetSolarLuminance();
//        }
//        radiance = mix(radiance, ground_radiance, ground_alpha);
//        radiance = mix(radiance, sphere_radiance, sphere_alpha);
//        color.rgb =
//            pow(vec3(1.0) - exp(-radiance / white_point * exposure), vec3(1.0 / 2.2));
//        color.a = 1.0;
//    }
//    else if (display_texture == 1) {
//        vec2 texCoord = gl_FragCoord.xy / vec2(256.f, 64.f);
//        color = texture(transmittance_texture, texCoord);
//    }
//    else if (display_texture == 2) {
//        vec2 texCoord = gl_FragCoord.xy / vec2(64.f, 16.f);
//        color = texture(irradiance_texture, texCoord);
//    }
//    else if (display_texture == 3) {
//        vec3 texCoord = vec3(vec2(gl_FragCoord.xy / vec2(8.f * 32.f, 128.f)), scatter_slice * 1.f);
//        color = texture(scattering_texture, texCoord);
//    }
//}