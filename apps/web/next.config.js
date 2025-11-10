/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
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
