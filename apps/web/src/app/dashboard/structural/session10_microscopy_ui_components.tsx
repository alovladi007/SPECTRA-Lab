'use client'

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import {
  LineChart, Line, ScatterChart, Scatter, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, ReferenceArea, Brush
} from 'recharts';
import {
  Activity, AlertCircle, Download, FileText, Play, Save,
  Upload, Zap, Eye, Layers, TrendingUp,
  Atom, Maximize, Target, Info, CheckCircle,
  ScanLine, Microscope, Grid3x3,
  BarChart3, Ruler, Database,
  Camera, Aperture, ZoomIn, ZoomOut,
  Circle, Square, PenTool, Crosshair
} from 'lucide-react';
import * as THREE from 'three';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid as ThreeGrid } from '@react-three/drei';

// Type definitions
interface MicroscopyImage {
  id: string;
  type: 'SEM' | 'TEM' | 'AFM';
  mode: string;
  data: number[][];
  metadata: {
    pixelSize: number;
    magnification?: number;
    voltage?: number;
    workingDistance?: number;
    scanSize?: [number, number];
  };
  timestamp: Date;
}

interface Particle {
  id: number;
  diameter: number;
  circularity: number;
  area: number;
}

// Image Viewer Component
const ImageViewer: React.FC<{
  image: number[][];
  colormap?: string;
  onRegionSelect?: (region: [number, number, number, number]) => void;
}> = ({ image, colormap = 'grayscale', onRegionSelect }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [selecting, setSelecting] = useState(false);
  const [selection, setSelection] = useState<[number, number, number, number] | null>(null);

  useEffect(() => {
    if (!canvasRef.current || !image) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw image
    const imageData = ctx.createImageData(image[0].length, image.length);

    for (let y = 0; y < image.length; y++) {
      for (let x = 0; x < image[0].length; x++) {
        const idx = (y * image[0].length + x) * 4;
        const value = image[y][x] * 255;

        if (colormap === 'grayscale') {
          imageData.data[idx] = value;
          imageData.data[idx + 1] = value;
          imageData.data[idx + 2] = value;
        } else if (colormap === 'viridis') {
          imageData.data[idx] = value * 0.3;
          imageData.data[idx + 1] = value * 0.7;
          imageData.data[idx + 2] = value * 0.5;
        }
        imageData.data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);

    // Draw selection
    if (selection) {
      ctx.strokeStyle = 'yellow';
      ctx.lineWidth = 2;
      ctx.strokeRect(...selection);
    }
  }, [image, colormap, zoom, pan, selection]);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(prev => Math.max(0.1, Math.min(10, prev * delta)));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.shiftKey) {
      setSelecting(true);
      const rect = canvasRef.current?.getBoundingClientRect();
      if (rect) {
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        setSelection([x, y, 0, 0]);
      }
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (selecting && selection) {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (rect) {
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        setSelection([selection[0], selection[1], x - selection[0], y - selection[1]]);
      }
    }
  };

  const handleMouseUp = () => {
    if (selecting && selection && onRegionSelect) {
      onRegionSelect(selection);
    }
    setSelecting(false);
  };

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={image?.[0]?.length || 512}
        height={image?.length || 512}
        className="border rounded cursor-crosshair"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
      />
      <div className="absolute top-2 right-2 flex gap-2">
        <Button size="sm" variant="secondary" onClick={() => setZoom(zoom * 1.2)}>
          <ZoomIn className="w-4 h-4" />
        </Button>
        <Button size="sm" variant="secondary" onClick={() => setZoom(zoom * 0.8)}>
          <ZoomOut className="w-4 h-4" />
        </Button>
        <Button size="sm" variant="secondary" onClick={() => setZoom(1)}>
          1:1
        </Button>
      </div>
      <div className="absolute bottom-2 left-2">
        <Badge>{Math.round(zoom * 100)}%</Badge>
      </div>
    </div>
  );
};

// 3D Surface Mesh Component
const Surface3DMesh: React.FC<{ heightMap: number[][], scanSize: [number, number] }> = ({ heightMap, scanSize }) => {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.z += 0.001;
    }
  });

  const geometry = useMemo(() => {
    const geo = new THREE.PlaneGeometry(
      scanSize[0] / 100,
      scanSize[1] / 100,
      heightMap[0].length - 1,
      heightMap.length - 1
    );

    const vertices = geo.attributes.position.array as Float32Array;
    for (let y = 0; y < heightMap.length; y++) {
      for (let x = 0; x < heightMap[0].length; x++) {
        const idx = (y * heightMap[0].length + x) * 3;
        vertices[idx + 2] = heightMap[y][x] / 100;
      }
    }

    geo.computeVertexNormals();
    return geo;
  }, [heightMap, scanSize]);

  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshStandardMaterial
        color="#8884d8"
        roughness={0.4}
        metalness={0.3}
        wireframe={false}
      />
    </mesh>
  );
};

// 3D Surface Viewer for AFM
const Surface3DViewer: React.FC<{ heightMap: number[][], scanSize: [number, number] }> = ({ heightMap, scanSize }) => {
  return (
    <Canvas camera={{ position: [0, -5, 5], fov: 50 }}>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <Surface3DMesh heightMap={heightMap} scanSize={scanSize} />
      <OrbitControls enableDamping />
      <ThreeGrid args={[10, 10]} />
    </Canvas>
  );
};

// ===========================
// SEM Interface
// ===========================
export const SEMInterface: React.FC = () => {
  const [activeTab, setActiveTab] = useState('acquisition');
  const [currentImage, setCurrentImage] = useState<MicroscopyImage | null>(null);
  const [isAcquiring, setIsAcquiring] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<any>(null);

  const [semParams, setSemParams] = useState({
    voltage: 15,
    current: 1,
    workingDistance: 10,
    magnification: 10000,
    detector: 'SE',
    scanSpeed: 'normal'
  });

  const loadDemoData = useCallback(() => {
    const size = 512;
    const imageData = Array.from({ length: size }, () =>
      Array.from({ length: size }, () => Math.random())
    );

    setCurrentImage({
      id: 'sem_demo_' + Date.now(),
      type: 'SEM',
      mode: 'SE',
      data: imageData,
      metadata: {
        pixelSize: 5,
        magnification: 10000,
        voltage: 15,
        workingDistance: 10
      },
      timestamp: new Date()
    });

    setAnalysisResults({
      particles: Array.from({ length: 20 }, (_, i) => ({
        id: i,
        diameter: 20 + Math.random() * 30,
        circularity: 0.7 + Math.random() * 0.3,
        area: Math.random() * 1000
      }))
    });
  }, []);

  const startAcquisition = useCallback(async () => {
    setIsAcquiring(true);
    await new Promise(resolve => setTimeout(resolve, 3000));
    loadDemoData();
    setIsAcquiring(false);
    setActiveTab('analysis');
  }, [loadDemoData]);

  const particleSizeData = useMemo(() => {
    if (!analysisResults?.particles) return [];

    const bins = Array.from({ length: 10 }, (_, i) => ({
      range: `${20 + i * 5}-${25 + i * 5}`,
      count: 0
    }));

    analysisResults.particles.forEach((p: any) => {
      const binIdx = Math.floor((p.diameter - 20) / 5);
      if (binIdx >= 0 && binIdx < 10) {
        bins[binIdx].count++;
      }
    });

    return bins;
  }, [analysisResults]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Camera className="w-6 h-6" />
                Scanning Electron Microscopy (SEM)
              </CardTitle>
              <CardDescription>
                High-resolution surface imaging and morphology analysis
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Badge variant={currentImage ? 'success' : 'secondary'}>
                {currentImage ? 'Image Acquired' : 'No Image'}
              </Badge>
              {isAcquiring && <Badge variant="warning">Acquiring...</Badge>}
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-4 w-full">
          <TabsTrigger value="acquisition">Acquisition</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
          <TabsTrigger value="measurements">Measurements</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
        </TabsList>

        {/* Acquisition Tab */}
        <TabsContent value="acquisition">
          <div className="grid grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">SEM Parameters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Accelerating Voltage (kV)</Label>
                  <input
                    type="range"
                    value={semParams.voltage}
                    onChange={(e) => setSemParams({ ...semParams, voltage: parseFloat(e.target.value) })}
                    min={1}
                    max={30}
                    step={0.5}
                    className="w-full"
                  />
                  <span className="text-sm text-muted-foreground">{semParams.voltage} kV</span>
                </div>

                <div className="space-y-2">
                  <Label>Beam Current (nA)</Label>
                  <input
                    type="range"
                    value={semParams.current}
                    onChange={(e) => setSemParams({ ...semParams, current: parseFloat(e.target.value) })}
                    min={0.1}
                    max={10}
                    step={0.1}
                    className="w-full"
                  />
                  <span className="text-sm text-muted-foreground">{semParams.current} nA</span>
                </div>

                <div className="space-y-2">
                  <Label>Working Distance (mm)</Label>
                  <input
                    type="range"
                    value={semParams.workingDistance}
                    onChange={(e) => setSemParams({ ...semParams, workingDistance: parseFloat(e.target.value) })}
                    min={3}
                    max={25}
                    step={0.5}
                    className="w-full"
                  />
                  <span className="text-sm text-muted-foreground">{semParams.workingDistance} mm</span>
                </div>

                <div className="space-y-2">
                  <Label>Detector</Label>
                  <Select
                    value={semParams.detector}
                    onValueChange={(v) => setSemParams({ ...semParams, detector: v })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="SE">Secondary Electron</SelectItem>
                      <SelectItem value="BSE">Backscattered Electron</SelectItem>
                      <SelectItem value="InLens">In-Lens</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Button
                  onClick={startAcquisition}
                  disabled={isAcquiring}
                  className="w-full"
                >
                  {isAcquiring ? (
                    <>
                      <Activity className="w-4 h-4 mr-2 animate-spin" />
                      Acquiring...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Start Acquisition
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Live Preview</CardTitle>
              </CardHeader>
              <CardContent>
                {currentImage ? (
                  <ImageViewer image={currentImage.data} />
                ) : (
                  <div className="h-[400px] bg-muted rounded flex items-center justify-center">
                    <div className="text-center text-muted-foreground">
                      <Microscope className="w-16 h-16 mx-auto mb-4 opacity-20" />
                      <p>No image acquired</p>
                      <Button
                        variant="outline"
                        onClick={loadDemoData}
                        className="mt-4"
                      >
                        Load Demo Data
                      </Button>
                    </div>
                  </div>
                )}

                {currentImage && (
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    <Card>
                      <CardContent className="pt-4">
                        <div className="text-sm text-muted-foreground">Field of View</div>
                        <div className="text-lg font-bold">
                          {(currentImage.data[0].length * currentImage.metadata.pixelSize).toFixed(0)} × {(currentImage.data.length * currentImage.metadata.pixelSize).toFixed(0)} nm
                        </div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-4">
                        <div className="text-sm text-muted-foreground">Magnification</div>
                        <div className="text-lg font-bold">
                          {currentImage.metadata.magnification?.toLocaleString()}×
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Analysis Tab */}
        <TabsContent value="analysis">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Particle Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              {analysisResults?.particles ? (
                <>
                  <div className="grid grid-cols-3 gap-4 mb-4">
                    <Card>
                      <CardContent className="pt-4">
                        <div className="text-2xl font-bold">
                          {analysisResults.particles.length}
                        </div>
                        <p className="text-xs text-muted-foreground">Particles Detected</p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-4">
                        <div className="text-2xl font-bold">
                          {(analysisResults.particles.reduce((sum: number, p: any) => sum + p.diameter, 0) / analysisResults.particles.length).toFixed(1)}
                        </div>
                        <p className="text-xs text-muted-foreground">Mean Diameter (nm)</p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-4">
                        <div className="text-2xl font-bold">
                          {(analysisResults.particles.reduce((sum: number, p: any) => sum + p.circularity, 0) / analysisResults.particles.length).toFixed(2)}
                        </div>
                        <p className="text-xs text-muted-foreground">Mean Circularity</p>
                      </CardContent>
                    </Card>
                  </div>

                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={particleSizeData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="range" label={{ value: 'Diameter Range (nm)', position: 'insideBottom', offset: -5 }} />
                      <YAxis label={{ value: 'Count', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Bar dataKey="count" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                </>
              ) : (
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>No Analysis</AlertTitle>
                  <AlertDescription>
                    Acquire an image first, then run particle detection
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Measurements Tab */}
        <TabsContent value="measurements">
          <div className="grid grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Measurement Tools</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button variant="outline" className="w-full justify-start">
                  <Ruler className="w-4 h-4 mr-2" />
                  Distance
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Circle className="w-4 h-4 mr-2" />
                  Area
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Square className="w-4 h-4 mr-2" />
                  Rectangle ROI
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <PenTool className="w-4 h-4 mr-2" />
                  Profile Line
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Crosshair className="w-4 h-4 mr-2" />
                  Point Analysis
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Recent Measurements</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between p-2 bg-muted rounded">
                    <span>Distance 1</span>
                    <span className="font-mono">125.3 nm</span>
                  </div>
                  <div className="flex justify-between p-2 bg-muted rounded">
                    <span>Area 1</span>
                    <span className="font-mono">3,542 nm²</span>
                  </div>
                  <div className="flex justify-between p-2 bg-muted rounded">
                    <span>Particle Count</span>
                    <span className="font-mono">20</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Results Tab */}
        <TabsContent value="results">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">Analysis Report</CardTitle>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline">
                    <Save className="w-4 h-4 mr-2" />
                    Save
                  </Button>
                  <Button size="sm" variant="outline">
                    <FileText className="w-4 h-4 mr-2" />
                    Export PDF
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold mb-2">Sample Information</h3>
                  <dl className="grid grid-cols-2 gap-2 text-sm">
                    <dt className="text-muted-foreground">Sample ID:</dt>
                    <dd>SEM-DEMO-001</dd>
                    <dt className="text-muted-foreground">Technique:</dt>
                    <dd>Scanning Electron Microscopy</dd>
                    <dt className="text-muted-foreground">Date:</dt>
                    <dd>{new Date().toLocaleDateString()}</dd>
                    <dt className="text-muted-foreground">Operator:</dt>
                    <dd>Lab User</dd>
                  </dl>
                </div>

                <Separator />

                <div>
                  <h3 className="font-semibold mb-2">Key Results</h3>
                  <ul className="space-y-1 text-sm">
                    <li>• Particle count: 20</li>
                    <li>• Mean diameter: 35.2 ± 8.4 nm</li>
                    <li>• Circularity: 0.85 ± 0.12</li>
                    <li>• Size uniformity: 76%</li>
                  </ul>
                </div>

                <Separator />

                <Alert>
                  <CheckCircle className="h-4 w-4" />
                  <AlertTitle>Analysis Complete</AlertTitle>
                  <AlertDescription>
                    All measurements have been processed successfully
                  </AlertDescription>
                </Alert>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

// ===========================
// TEM Interface
// ===========================
export const TEMInterface: React.FC = () => {
  const [activeTab, setActiveTab] = useState('acquisition');
  const [currentImage, setCurrentImage] = useState<MicroscopyImage | null>(null);
  const [isAcquiring, setIsAcquiring] = useState(false);

  const [temParams, setTemParams] = useState({
    voltage: 200,
    mode: 'BF',
    magnification: 50000,
    defocus: 0,
    objectiveAperture: 40
  });

  const loadDemoData = useCallback(() => {
    const size = 512;
    const imageData = Array.from({ length: size }, () =>
      Array.from({ length: size }, () => Math.random())
    );

    setCurrentImage({
      id: 'tem_demo_' + Date.now(),
      type: 'TEM',
      mode: 'BF',
      data: imageData,
      metadata: {
        pixelSize: 0.1,
        magnification: 50000,
        voltage: 200
      },
      timestamp: new Date()
    });
  }, []);

  const startAcquisition = useCallback(async () => {
    setIsAcquiring(true);
    await new Promise(resolve => setTimeout(resolve, 3000));
    loadDemoData();
    setIsAcquiring(false);
    setActiveTab('analysis');
  }, [loadDemoData]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Aperture className="w-6 h-6" />
                Transmission Electron Microscopy (TEM)
              </CardTitle>
              <CardDescription>
                High-resolution internal structure and diffraction analysis
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Badge variant={currentImage ? 'success' : 'secondary'}>
                {currentImage ? 'Image Acquired' : 'No Image'}
              </Badge>
              {isAcquiring && <Badge variant="warning">Acquiring...</Badge>}
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-4 w-full">
          <TabsTrigger value="acquisition">Acquisition</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
          <TabsTrigger value="measurements">Measurements</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
        </TabsList>

        {/* Acquisition Tab */}
        <TabsContent value="acquisition">
          <div className="grid grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">TEM Parameters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Accelerating Voltage (kV)</Label>
                  <Select
                    value={temParams.voltage.toString()}
                    onValueChange={(v) => setTemParams({ ...temParams, voltage: parseInt(v) })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="80">80 kV</SelectItem>
                      <SelectItem value="120">120 kV</SelectItem>
                      <SelectItem value="200">200 kV</SelectItem>
                      <SelectItem value="300">300 kV</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Imaging Mode</Label>
                  <Select
                    value={temParams.mode}
                    onValueChange={(v) => setTemParams({ ...temParams, mode: v })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="BF">Bright Field</SelectItem>
                      <SelectItem value="DF">Dark Field</SelectItem>
                      <SelectItem value="HRTEM">High Resolution</SelectItem>
                      <SelectItem value="SAED">Diffraction</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Objective Aperture (μm)</Label>
                  <input
                    type="range"
                    value={temParams.objectiveAperture}
                    onChange={(e) => setTemParams({ ...temParams, objectiveAperture: parseFloat(e.target.value) })}
                    min={10}
                    max={100}
                    step={10}
                    className="w-full"
                  />
                  <span className="text-sm text-muted-foreground">{temParams.objectiveAperture} μm</span>
                </div>

                <Button
                  onClick={startAcquisition}
                  disabled={isAcquiring}
                  className="w-full"
                >
                  {isAcquiring ? (
                    <>
                      <Activity className="w-4 h-4 mr-2 animate-spin" />
                      Acquiring...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Start Acquisition
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Live Preview</CardTitle>
              </CardHeader>
              <CardContent>
                {currentImage ? (
                  <ImageViewer image={currentImage.data} />
                ) : (
                  <div className="h-[400px] bg-muted rounded flex items-center justify-center">
                    <div className="text-center text-muted-foreground">
                      <Microscope className="w-16 h-16 mx-auto mb-4 opacity-20" />
                      <p>No image acquired</p>
                      <Button
                        variant="outline"
                        onClick={loadDemoData}
                        className="mt-4"
                      >
                        Load Demo Data
                      </Button>
                    </div>
                  </div>
                )}

                {currentImage && (
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    <Card>
                      <CardContent className="pt-4">
                        <div className="text-sm text-muted-foreground">Pixel Size</div>
                        <div className="text-lg font-bold">
                          {currentImage.metadata.pixelSize.toFixed(2)} nm/px
                        </div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-4">
                        <div className="text-sm text-muted-foreground">Voltage</div>
                        <div className="text-lg font-bold">
                          {currentImage.metadata.voltage} kV
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Analysis Tab */}
        <TabsContent value="analysis">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Diffraction Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold mb-2">FFT Pattern</h4>
                  <div className="h-[200px] bg-muted rounded flex items-center justify-center">
                    <p className="text-sm text-muted-foreground">FFT visualization would appear here</p>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">d-Spacings</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between p-2 bg-muted rounded">
                      <span>d₁:</span>
                      <span className="font-mono">3.136 Å</span>
                    </div>
                    <div className="flex justify-between p-2 bg-muted rounded">
                      <span>d₂:</span>
                      <span className="font-mono">1.920 Å</span>
                    </div>
                    <div className="flex justify-between p-2 bg-muted rounded">
                      <span>d₃:</span>
                      <span className="font-mono">1.638 Å</span>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Measurements Tab */}
        <TabsContent value="measurements">
          <div className="grid grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Measurement Tools</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button variant="outline" className="w-full justify-start">
                  <Ruler className="w-4 h-4 mr-2" />
                  Lattice Spacing
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Circle className="w-4 h-4 mr-2" />
                  Particle Size
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Atom className="w-4 h-4 mr-2" />
                  Interplanar Distance
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Recent Measurements</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between p-2 bg-muted rounded">
                    <span>d-spacing 1</span>
                    <span className="font-mono">3.14 Å</span>
                  </div>
                  <div className="flex justify-between p-2 bg-muted rounded">
                    <span>Particle 1</span>
                    <span className="font-mono">25.3 nm</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Results Tab */}
        <TabsContent value="results">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">Analysis Report</CardTitle>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline">
                    <Save className="w-4 h-4 mr-2" />
                    Save
                  </Button>
                  <Button size="sm" variant="outline">
                    <FileText className="w-4 h-4 mr-2" />
                    Export PDF
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold mb-2">Sample Information</h3>
                  <dl className="grid grid-cols-2 gap-2 text-sm">
                    <dt className="text-muted-foreground">Sample ID:</dt>
                    <dd>TEM-DEMO-001</dd>
                    <dt className="text-muted-foreground">Technique:</dt>
                    <dd>Transmission Electron Microscopy</dd>
                    <dt className="text-muted-foreground">Date:</dt>
                    <dd>{new Date().toLocaleDateString()}</dd>
                    <dt className="text-muted-foreground">Operator:</dt>
                    <dd>Lab User</dd>
                  </dl>
                </div>

                <Separator />

                <div>
                  <h3 className="font-semibold mb-2">Key Results</h3>
                  <ul className="space-y-1 text-sm">
                    <li>• Crystal structure: FCC</li>
                    <li>• Lattice parameter: 3.615 Å</li>
                    <li>• Defect density: Low</li>
                    <li>• Sample thickness: ~50 nm</li>
                  </ul>
                </div>

                <Separator />

                <Alert>
                  <CheckCircle className="h-4 w-4" />
                  <AlertTitle>Analysis Complete</AlertTitle>
                  <AlertDescription>
                    All measurements have been processed successfully
                  </AlertDescription>
                </Alert>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

// ===========================
// AFM Interface
// ===========================
export const AFMInterface: React.FC = () => {
  const [activeTab, setActiveTab] = useState('acquisition');
  const [currentImage, setCurrentImage] = useState<MicroscopyImage | null>(null);
  const [isAcquiring, setIsAcquiring] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<any>(null);

  const [afmParams, setAfmParams] = useState({
    mode: 'tapping',
    scanSize: [1000, 1000] as [number, number],
    scanRate: 1,
    setPoint: 0.5,
    gain: [1, 1],
    resolution: 256
  });

  const loadDemoData = useCallback(() => {
    const size = 256;
    const imageData = Array.from({ length: size }, () =>
      Array.from({ length: size }, () => Math.random())
    );

    setCurrentImage({
      id: 'afm_demo_' + Date.now(),
      type: 'AFM',
      mode: 'height',
      data: imageData,
      metadata: {
        pixelSize: 4,
        scanSize: [1000, 1000]
      },
      timestamp: new Date()
    });

    setAnalysisResults({
      roughness: {
        Sa: 2.5 + Math.random(),
        Sq: 3.2 + Math.random(),
        Sp: 8.5 + Math.random(),
        Sv: 7.2 + Math.random()
      }
    });
  }, []);

  const startAcquisition = useCallback(async () => {
    setIsAcquiring(true);
    await new Promise(resolve => setTimeout(resolve, 3000));
    loadDemoData();
    setIsAcquiring(false);
    setActiveTab('analysis');
  }, [loadDemoData]);

  const roughnessData = useMemo(() => {
    if (!analysisResults?.roughness) return [];

    return Object.entries(analysisResults.roughness).map(([key, value]) => ({
      parameter: key,
      value: value as number
    }));
  }, [analysisResults]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Grid3x3 className="w-6 h-6" />
                Atomic Force Microscopy (AFM)
              </CardTitle>
              <CardDescription>
                3D surface topography and nanoscale roughness analysis
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Badge variant={currentImage ? 'success' : 'secondary'}>
                {currentImage ? 'Scan Complete' : 'No Scan'}
              </Badge>
              {isAcquiring && <Badge variant="warning">Scanning...</Badge>}
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-4 w-full">
          <TabsTrigger value="acquisition">Acquisition</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
          <TabsTrigger value="measurements">Measurements</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
        </TabsList>

        {/* Acquisition Tab */}
        <TabsContent value="acquisition">
          <div className="grid grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">AFM Parameters</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Scanning Mode</Label>
                  <Select
                    value={afmParams.mode}
                    onValueChange={(v) => setAfmParams({ ...afmParams, mode: v })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="contact">Contact</SelectItem>
                      <SelectItem value="tapping">Tapping</SelectItem>
                      <SelectItem value="non-contact">Non-Contact</SelectItem>
                      <SelectItem value="phase">Phase Imaging</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Scan Size (nm)</Label>
                  <div className="grid grid-cols-2 gap-2">
                    <Input
                      type="number"
                      value={afmParams.scanSize[0]}
                      onChange={(e) => setAfmParams({
                        ...afmParams,
                        scanSize: [parseInt(e.target.value), afmParams.scanSize[1]]
                      })}
                    />
                    <Input
                      type="number"
                      value={afmParams.scanSize[1]}
                      onChange={(e) => setAfmParams({
                        ...afmParams,
                        scanSize: [afmParams.scanSize[0], parseInt(e.target.value)]
                      })}
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Scan Rate (Hz)</Label>
                  <input
                    type="range"
                    value={afmParams.scanRate}
                    onChange={(e) => setAfmParams({ ...afmParams, scanRate: parseFloat(e.target.value) })}
                    min={0.1}
                    max={2}
                    step={0.1}
                    className="w-full"
                  />
                  <span className="text-sm text-muted-foreground">{afmParams.scanRate} Hz</span>
                </div>

                <Button
                  onClick={startAcquisition}
                  disabled={isAcquiring}
                  className="w-full"
                >
                  {isAcquiring ? (
                    <>
                      <Activity className="w-4 h-4 mr-2 animate-spin" />
                      Scanning...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      Start Scan
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Live Preview</CardTitle>
              </CardHeader>
              <CardContent>
                {currentImage ? (
                  <ImageViewer image={currentImage.data} colormap="viridis" />
                ) : (
                  <div className="h-[400px] bg-muted rounded flex items-center justify-center">
                    <div className="text-center text-muted-foreground">
                      <Grid3x3 className="w-16 h-16 mx-auto mb-4 opacity-20" />
                      <p>No scan acquired</p>
                      <Button
                        variant="outline"
                        onClick={loadDemoData}
                        className="mt-4"
                      >
                        Load Demo Data
                      </Button>
                    </div>
                  </div>
                )}

                {currentImage && (
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    <Card>
                      <CardContent className="pt-4">
                        <div className="text-sm text-muted-foreground">Scan Area</div>
                        <div className="text-lg font-bold">
                          {currentImage.metadata.scanSize![0]} × {currentImage.metadata.scanSize![1]} nm
                        </div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-4">
                        <div className="text-sm text-muted-foreground">Resolution</div>
                        <div className="text-lg font-bold">
                          {currentImage.data[0].length} × {currentImage.data.length} px
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Analysis Tab */}
        <TabsContent value="analysis">
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">3D Surface Visualization</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-[400px]">
                  {currentImage ? (
                    <Surface3DViewer
                      heightMap={currentImage.data}
                      scanSize={afmParams.scanSize}
                    />
                  ) : (
                    <div className="h-full bg-muted rounded flex items-center justify-center">
                      <p className="text-sm text-muted-foreground">No data to visualize</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Roughness Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                {analysisResults?.roughness ? (
                  <div className="grid grid-cols-2 gap-4">
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={roughnessData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="parameter" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="value" fill="#82ca9d" />
                      </BarChart>
                    </ResponsiveContainer>

                    <div className="space-y-2">
                      <div className="flex justify-between p-2 bg-muted rounded">
                        <span>Sa (Average):</span>
                        <span className="font-bold">{analysisResults.roughness.Sa.toFixed(2)} nm</span>
                      </div>
                      <div className="flex justify-between p-2 bg-muted rounded">
                        <span>Sq (RMS):</span>
                        <span className="font-bold">{analysisResults.roughness.Sq.toFixed(2)} nm</span>
                      </div>
                      <div className="flex justify-between p-2 bg-muted rounded">
                        <span>Sp (Peak):</span>
                        <span className="font-bold">{analysisResults.roughness.Sp.toFixed(2)} nm</span>
                      </div>
                      <div className="flex justify-between p-2 bg-muted rounded">
                        <span>Sv (Valley):</span>
                        <span className="font-bold">{analysisResults.roughness.Sv.toFixed(2)} nm</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <Alert>
                    <Info className="h-4 w-4" />
                    <AlertDescription>
                      No roughness data available. Acquire a scan first.
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Measurements Tab */}
        <TabsContent value="measurements">
          <div className="grid grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Measurement Tools</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button variant="outline" className="w-full justify-start">
                  <Ruler className="w-4 h-4 mr-2" />
                  Height Profile
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Square className="w-4 h-4 mr-2" />
                  Region Roughness
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <PenTool className="w-4 h-4 mr-2" />
                  Line Profile
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Recent Measurements</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between p-2 bg-muted rounded">
                    <span>Max Height</span>
                    <span className="font-mono">8.7 nm</span>
                  </div>
                  <div className="flex justify-between p-2 bg-muted rounded">
                    <span>Avg Roughness</span>
                    <span className="font-mono">2.5 nm</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Results Tab */}
        <TabsContent value="results">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">Analysis Report</CardTitle>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline">
                    <Save className="w-4 h-4 mr-2" />
                    Save
                  </Button>
                  <Button size="sm" variant="outline">
                    <FileText className="w-4 h-4 mr-2" />
                    Export PDF
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold mb-2">Sample Information</h3>
                  <dl className="grid grid-cols-2 gap-2 text-sm">
                    <dt className="text-muted-foreground">Sample ID:</dt>
                    <dd>AFM-DEMO-001</dd>
                    <dt className="text-muted-foreground">Technique:</dt>
                    <dd>Atomic Force Microscopy</dd>
                    <dt className="text-muted-foreground">Date:</dt>
                    <dd>{new Date().toLocaleDateString()}</dd>
                    <dt className="text-muted-foreground">Operator:</dt>
                    <dd>Lab User</dd>
                  </dl>
                </div>

                <Separator />

                <div>
                  <h3 className="font-semibold mb-2">Key Results</h3>
                  <ul className="space-y-1 text-sm">
                    <li>• Sa roughness: 2.5 nm</li>
                    <li>• Sq roughness: 3.2 nm</li>
                    <li>• Peak-to-valley: 15.7 nm</li>
                    <li>• Grain size: 45 ± 12 nm</li>
                  </ul>
                </div>

                <Separator />

                <Alert>
                  <CheckCircle className="h-4 w-4" />
                  <AlertTitle>Analysis Complete</AlertTitle>
                  <AlertDescription>
                    All measurements have been processed successfully
                  </AlertDescription>
                </Alert>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

// ===========================
// Optical Microscopy Interface
// ===========================
export const OpticalMicroscopyInterface: React.FC = () => {
  const [activeTab, setActiveTab] = useState('acquisition');
  const [currentImage, setCurrentImage] = useState<MicroscopyImage | null>(null);
  const [isAcquiring, setIsAcquiring] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<any>(null);

  const [params, setParams] = useState({
    objective: '40x',
    illumination: 'bright-field',
    aperture: 0.75,
    filterSet: 'standard'
  });

  const loadDemoData = useCallback(() => {
    const size = 512;
    const imageData = Array.from({ length: size }, () =>
      Array.from({ length: size }, () => Math.random())
    );

    setCurrentImage({
      id: 'optical_demo_' + Date.now(),
      type: 'SEM',
      mode: 'bright-field',
      data: imageData,
      metadata: {
        pixelSize: 0.5,
        magnification: 400
      },
      timestamp: new Date()
    });

    setAnalysisResults({
      features: Array.from({ length: 15 }, (_, i) => ({
        id: i,
        size: 10 + Math.random() * 20,
        intensity: 0.3 + Math.random() * 0.7
      }))
    });
  }, []);

  const startAcquisition = useCallback(async () => {
    setIsAcquiring(true);
    await new Promise(resolve => setTimeout(resolve, 2000));
    loadDemoData();
    setIsAcquiring(false);
    setActiveTab('analysis');
  }, [loadDemoData]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Microscope className="w-6 h-6" />
                Optical Microscopy
              </CardTitle>
              <CardDescription>
                High-magnification optical imaging for surface features and defects
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Badge variant={currentImage ? 'success' : 'secondary'}>
                {currentImage ? 'Image Captured' : 'No Image'}
              </Badge>
              {isAcquiring && <Badge variant="warning">Acquiring...</Badge>}
            </div>
          </div>
        </CardHeader>
      </Card>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-4 w-full">
          <TabsTrigger value="acquisition">Acquisition</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
          <TabsTrigger value="measurements">Measurements</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
        </TabsList>

        <TabsContent value="acquisition">
          <div className="grid grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Microscope Settings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Objective</Label>
                  <Select value={params.objective} onValueChange={(v) => setParams({ ...params, objective: v })}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="5x">5× (NA 0.15)</SelectItem>
                      <SelectItem value="10x">10× (NA 0.30)</SelectItem>
                      <SelectItem value="20x">20× (NA 0.50)</SelectItem>
                      <SelectItem value="40x">40× (NA 0.75)</SelectItem>
                      <SelectItem value="100x">100× (NA 1.40 Oil)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Illumination Mode</Label>
                  <Select value={params.illumination} onValueChange={(v) => setParams({ ...params, illumination: v })}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="bright-field">Bright Field</SelectItem>
                      <SelectItem value="dark-field">Dark Field</SelectItem>
                      <SelectItem value="phase-contrast">Phase Contrast</SelectItem>
                      <SelectItem value="dic">DIC (Nomarski)</SelectItem>
                      <SelectItem value="polarized">Polarized Light</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Button onClick={startAcquisition} disabled={isAcquiring} className="w-full">
                  {isAcquiring ? (
                    <><Activity className="w-4 h-4 mr-2 animate-spin" />Capturing...</>
                  ) : (
                    <><Play className="w-4 h-4 mr-2" />Capture Image</>
                  )}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Live View</CardTitle>
              </CardHeader>
              <CardContent>
                {currentImage ? (
                  <ImageViewer image={currentImage.data} />
                ) : (
                  <div className="h-[400px] bg-muted rounded flex items-center justify-center">
                    <div className="text-center text-muted-foreground">
                      <Microscope className="w-16 h-16 mx-auto mb-4 opacity-20" />
                      <p>No image captured</p>
                      <Button variant="outline" onClick={loadDemoData} className="mt-4">
                        Load Demo Data
                      </Button>
                    </div>
                  </div>
                )}

                {currentImage && (
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    <Card>
                      <CardContent className="pt-4">
                        <div className="text-sm text-muted-foreground">Magnification</div>
                        <div className="text-lg font-bold">{params.objective}</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-4">
                        <div className="text-sm text-muted-foreground">Resolution</div>
                        <div className="text-lg font-bold">{currentImage.metadata.pixelSize.toFixed(2)} μm/px</div>
                      </CardContent>
                    </Card>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="analysis">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Feature Detection</CardTitle>
            </CardHeader>
            <CardContent>
              {analysisResults?.features ? (
                <>
                  <div className="grid grid-cols-3 gap-4 mb-4">
                    <Card>
                      <CardContent className="pt-4">
                        <div className="text-2xl font-bold">{analysisResults.features.length}</div>
                        <p className="text-xs text-muted-foreground">Features Detected</p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-4">
                        <div className="text-2xl font-bold">
                          {(analysisResults.features.reduce((sum: number, f: any) => sum + f.size, 0) / analysisResults.features.length).toFixed(1)}
                        </div>
                        <p className="text-xs text-muted-foreground">Mean Size (μm)</p>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-4">
                        <div className="text-2xl font-bold">
                          {(analysisResults.features.reduce((sum: number, f: any) => sum + f.intensity, 0) / analysisResults.features.length).toFixed(2)}
                        </div>
                        <p className="text-xs text-muted-foreground">Mean Intensity</p>
                      </CardContent>
                    </Card>
                  </div>
                  <div className="h-[200px] bg-muted rounded flex items-center justify-center">
                    <p className="text-sm text-muted-foreground">Feature visualization would appear here</p>
                  </div>
                </>
              ) : (
                <Alert>
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>No Analysis</AlertTitle>
                  <AlertDescription>Capture an image first, then run feature detection</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="measurements">
          <div className="grid grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Measurement Tools</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button variant="outline" className="w-full justify-start">
                  <Ruler className="w-4 h-4 mr-2" />Length Measurement
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Circle className="w-4 h-4 mr-2" />Area Measurement
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Square className="w-4 h-4 mr-2" />ROI Selection
                </Button>
                <Button variant="outline" className="w-full justify-start">
                  <Eye className="w-4 h-4 mr-2" />Intensity Profile
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Recent Measurements</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between p-2 bg-muted rounded">
                    <span>Feature 1</span>
                    <span className="font-mono">15.2 μm</span>
                  </div>
                  <div className="flex justify-between p-2 bg-muted rounded">
                    <span>Feature 2</span>
                    <span className="font-mono">22.8 μm</span>
                  </div>
                  <div className="flex justify-between p-2 bg-muted rounded">
                    <span>ROI Area</span>
                    <span className="font-mono">245 μm²</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="results">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">Analysis Report</CardTitle>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline">
                    <Save className="w-4 h-4 mr-2" />Save
                  </Button>
                  <Button size="sm" variant="outline">
                    <FileText className="w-4 h-4 mr-2" />Export PDF
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold mb-2">Sample Information</h3>
                  <dl className="grid grid-cols-2 gap-2 text-sm">
                    <dt className="text-muted-foreground">Sample ID:</dt>
                    <dd>OPT-DEMO-001</dd>
                    <dt className="text-muted-foreground">Technique:</dt>
                    <dd>Optical Microscopy</dd>
                    <dt className="text-muted-foreground">Date:</dt>
                    <dd>{new Date().toLocaleDateString()}</dd>
                    <dt className="text-muted-foreground">Operator:</dt>
                    <dd>Lab User</dd>
                  </dl>
                </div>

                <Separator />

                <div>
                  <h3 className="font-semibold mb-2">Key Results</h3>
                  <ul className="space-y-1 text-sm">
                    <li>• Features detected: 15</li>
                    <li>• Mean feature size: 15.3 μm</li>
                    <li>• Surface coverage: 12.5%</li>
                    <li>• Image quality: Excellent</li>
                  </ul>
                </div>

                <Separator />

                <Alert>
                  <CheckCircle className="h-4 w-4" />
                  <AlertTitle>Analysis Complete</AlertTitle>
                  <AlertDescription>
                    All measurements have been processed successfully
                  </AlertDescription>
                </Alert>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};
