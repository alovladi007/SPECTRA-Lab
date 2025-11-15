/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      // Analysis Service (Port 8001) - CVD, Diffusion, Oxidation, Calibration, Predictive Maintenance
      {
        source: '/api/v1/cvd/:path*',
        destination: 'http://localhost:8001/api/v1/cvd/:path*',
      },
      {
        source: '/api/v1/diffusion/:path*',
        destination: 'http://localhost:8001/api/v1/diffusion/:path*',
      },
      {
        source: '/api/v1/oxidation/:path*',
        destination: 'http://localhost:8001/api/v1/oxidation/:path*',
      },
      {
        source: '/api/v1/calibration/:path*',
        destination: 'http://localhost:8001/api/v1/calibration/:path*',
      },
      {
        source: '/api/v1/predictive-maintenance/:path*',
        destination: 'http://localhost:8001/api/v1/predictive-maintenance/:path*',
      },
      // Other services (Port 8003)
      {
        source: '/api/v1/implant/:path*',
        destination: 'http://localhost:8003/api/v1/implant/:path*',
      },
      {
        source: '/api/v1/rtp/:path*',
        destination: 'http://localhost:8003/api/v1/rtp/:path*',
      },
      {
        source: '/api/v1/spc/:path*',
        destination: 'http://localhost:8003/api/v1/spc/:path*',
      },
      {
        source: '/api/v1/vm/:path*',
        destination: 'http://localhost:8003/api/v1/vm/:path*',
      },
    ];
  },
}

module.exports = nextConfig
