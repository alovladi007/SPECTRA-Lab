'use client';

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Switch } from '@/components/ui/switch';
import { Checkbox } from '@/components/ui/checkbox';
import { Separator } from '@/components/ui/separator';
import { 
  LineChart, Line, ScatterChart, Scatter, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, ReferenceLine, ReferenceArea, Brush,
  ComposedChart, ErrorBar, Histogram, RadialBarChart, RadialBar,
  Heatmap, Contour
} from 'recharts';
import {
  Activity, AlertCircle, Download, FileText, Play, Save, Settings,
  Upload, Zap, Eye, Layers, TrendingUp, Thermometer, Gauge,
  Atom, Maximize, Target, Info, CheckCircle, Beaker,
  ScanLine, Lightbulb, Microscope, Grid3x3, Hexagon,
  BarChart3, Ruler, ArrowUpRight, ArrowDownRight, Database,
  Cpu, Camera, Aperture, Move, ZoomIn, ZoomOut, Crosshair,
  Scissors, Circle, Square, Triangle, PenTool, Eraser
} from 'lucide-react';
import * as THREE from 'three';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { OrbitControls, Grid, PerspectiveCamera } from '@react-three/drei';

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
  centroid: [number, number];
  area: number;
  perimeter: number;
  diameter: number;
  circularity: number;
  aspectRatio: number;
  orientation: number;
  intensityMean: number;
}

interface AFMData {
  height: number[][];
  amplitude?: number[][];
  phase?: number[][];
  forceCurves?: number[][];
  roughness?: {
    Sa: number;
    Sq: number;
    Sp: number;
    Sv: number;
    Ssk: number;
    Sku: number;
  };
}

interface MeasurementResult {
  type: string;
  value: number;
  unit: string;
  error?: number;
  confidence?: number;
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
          // Simplified viridis colormap
          imageData.data[idx] = value * 0.3;
          imageData.data[idx + 1] = value * 0.7;
          imageData.data[idx + 2] = value * 0.5;
        }
        imageData.data[idx + 3] = 255;
      }
    }
    
    ctx.putImageData(imageData, 0, 0);
    
    // Apply zoom and pan
    ctx.save();
    ctx.scale(zoom, zoom);
    ctx.translate(pan.x, pan.y);
    ctx.restore();
    
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

// 3D Surface Viewer for AFM
const Surface3DViewer: React.FC<{ heightMap: number[][], scanSize: [number, number] }> = ({ heightMap, scanSize }) => {
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
    <Canvas camera={{ position: [0, -5, 5], fov: 50 }}>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <mesh ref={meshRef} geometry={geometry}>
        <meshStandardMaterial 
          color="#8884d8" 
          roughness={0.4}
          metalness={0.3}
          wireframe={false}
        />
      </mesh>
      <OrbitControls enableDamping />
      <Grid args={[10, 10]} />
    </Canvas>
  );
};

// Main Microscopy Interface Component
const MicroscopyInterface: React.FC = () => {
  // State management
  const [activeTab, setActiveTab] = useState('acquisition');
  const [selectedTechnique, setSelectedTechnique] = useState<'SEM' | 'TEM' | 'AFM'>('SEM');
  const [currentImage, setCurrentImage] = useState<MicroscopyImage | null>(null);
  const [isAcquiring, setIsAcquiring] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  
  // SEM parameters
  const [semParams, setSemParams] = useState({
    voltage: 15,
    current: 1,
    workingDistance: 10,
    magnification: 10000,
    detector: 'SE',
    scanSpeed: 'normal'
  });
  
  // TEM parameters
  const [temParams, setTemParams] = useState({
    voltage: 200,
    mode: 'BF',
    magnification: 50000,
    defocus: 0,
    objectiveAperture: 40
  });
  
  // AFM parameters
  const [afmParams, setAfmParams] = useState({
    mode: 'tapping',
    scanSize: [1000, 1000],
    scanRate: 1,
    setPoint: 0.5,
    gain: [1, 1],
    resolution: 256
  });

  // Load demo data
  const loadDemoData = useCallback(() => {
    // Generate synthetic image based on technique
    const size = 512;
    const imageData = Array.from({ length: size }, () => 
      Array.from({ length: size }, () => Math.random())
    );
    
    setCurrentImage({
      id: 'demo_' + Date.now(),
      type: selectedTechnique,
      mode: selectedTechnique === 'SEM' ? 'SE' : selectedTechnique === 'TEM' ? 'BF' : 'height',
      data: imageData,
      metadata: {
        pixelSize: selectedTechnique === 'TEM' ? 0.1 : selectedTechnique === 'AFM' ? 4 : 5,
        magnification: selectedTechnique === 'SEM' ? 10000 : selectedTechnique === 'TEM' ? 50000 : undefined,
        voltage: selectedTechnique === 'AFM' ? undefined : selectedTechnique === 'SEM' ? 15 : 200,
        scanSize: selectedTechnique === 'AFM' ? [1000, 1000] : undefined
      },
      timestamp: new Date()
    });
    
    // Generate demo analysis results
    setAnalysisResults({
      particles: Array.from({ length: 20 }, (_, i) => ({
        id: i,
        diameter: 20 + Math.random() * 30,
        circularity: 0.7 + Math.random() * 0.3,
        area: Math.random() * 1000
      })),
      roughness: {
        Sa: 2.5 + Math.random(),
        Sq: 3.2 + Math.random(),
        Sp: 8.5 + Math.random(),
        Sv: 7.2 + Math.random()
      }
    });
  }, [selectedTechnique]);

  // Start acquisition
  const startAcquisition = useCallback(async () => {
    setIsAcquiring(true);
    
    // Simulate acquisition
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    loadDemoData();
    setIsAcquiring(false);
    setActiveTab('analysis');
  }, [loadDemoData]);

  // Particle size distribution chart data
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

  // Roughness parameters for chart
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
                <Microscope className="w-6 h-6" />
                Microscopy Analysis Suite
              </CardTitle>
              <CardDescription>
                SEM, TEM, and AFM imaging and analysis
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Badge variant={currentImage ? 'success' : 'secondary'}>
                {currentImage ? `${currentImage.type} Image` : 'No Image'}
              </Badge>
              {isAcquiring && <Badge variant="warning">Acquiring...</Badge>}
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Technique Selection */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex gap-4">
            <Button
              variant={selectedTechnique === 'SEM' ? 'default' : 'outline'}
              onClick={() => setSelectedTechnique('SEM')}
              className="flex-1"
            >
              <Camera className="w-4 h-4 mr-2" />
              SEM
            </Button>
            <Button
              variant={selectedTechnique === 'TEM' ? 'default' : 'outline'}
              onClick={() => setSelectedTechnique('TEM')}
              className="flex-1"
            >
              <Aperture className="w-4 h-4 mr-2" />
              TEM
            </Button>
            <Button
              variant={selectedTechnique === 'AFM' ? 'default' : 'outline'}
              onClick={() => setSelectedTechnique('AFM')}
              className="flex-1"
            >
              <Grid3x3 className="w-4 h-4 mr-2" />
              AFM
            </Button>
          </div>
        </CardContent>
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
            {/* Parameters Panel */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">
                  {selectedTechnique} Parameters
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {selectedTechnique === 'SEM' && (
                  <>
                    <div className="space-y-2">
                      <Label>Accelerating Voltage (kV)</Label>
                      <Slider
                        value={[semParams.voltage]}
                        onValueChange={(v) => setSemParams({ ...semParams, voltage: v[0] })}
                        min={1}
                        max={30}
                        step={0.5}
                      />
                      <span className="text-sm text-muted-foreground">{semParams.voltage} kV</span>
                    </div>
                    
                    <div className="space-y-2">
                      <Label>Beam Current (nA)</Label>
                      <Slider
                        value={[semParams.current]}
                        onValueChange={(v) => setSemParams({ ...semParams, current: v[0] })}
                        min={0.1}
                        max={10}
                        step={0.1}
                      />
                      <span className="text-sm text-muted-foreground">{semParams.current} nA</span>
                    </div>
                    
                    <div className="space-y-2">
                      <Label>Working Distance (mm)</Label>
                      <Slider
                        value={[semParams.workingDistance]}
                        onValueChange={(v) => setSemParams({ ...semParams, workingDistance: v[0] })}
                        min={3}
                        max={25}
                        step={0.5}
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
                  </>
                )}

                {selectedTechnique === 'TEM' && (
                  <>
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
                      <Slider
                        value={[temParams.objectiveAperture]}
                        onValueChange={(v) => setTemParams({ ...temParams, objectiveAperture: v[0] })}
                        min={10}
                        max={100}
                        step={10}
                      />
                      <span className="text-sm text-muted-foreground">{temParams.objectiveAperture} μm</span>
                    </div>
                  </>
                )}

                {selectedTechnique === 'AFM' && (
                  <>
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
                      <Slider
                        value={[afmParams.scanRate]}
                        onValueChange={(v) => setAfmParams({ ...afmParams, scanRate: v[0] })}
                        min={0.1}
                        max={2}
                        step={0.1}
                      />
                      <span className="text-sm text-muted-foreground">{afmParams.scanRate} Hz</span>
                    </div>
                  </>
                )}

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

            {/* Preview/Status Panel */}
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
                          {currentImage.metadata.scanSize ? 
                            `${currentImage.metadata.scanSize[0]} × ${currentImage.metadata.scanSize[1]} nm` :
                            `${(currentImage.data[0].length * currentImage.metadata.pixelSize).toFixed(0)} × ${(currentImage.data.length * currentImage.metadata.pixelSize).toFixed(0)} nm`
                          }
                        </div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="pt-4">
                        <div className="text-sm text-muted-foreground">Pixel Size</div>
                        <div className="text-lg font-bold">
                          {currentImage.metadata.pixelSize.toFixed(2)} nm/px
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
            {selectedTechnique === 'SEM' && (
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
                              {analysisResults.particles.reduce((sum: number, p: any) => sum + p.diameter, 0) / analysisResults.particles.length:.1f}
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
            )}

            {selectedTechnique === 'TEM' && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Diffraction Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-semibold mb-2">FFT Pattern</h4>
                      <div className="h-[200px] bg-muted rounded"></div>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-2">d-Spacings</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span>d₁:</span>
                          <span className="font-mono">3.136 Å</span>
                        </div>
                        <div className="flex justify-between">
                          <span>d₂:</span>
                          <span className="font-mono">1.920 Å</span>
                        </div>
                        <div className="flex justify-between">
                          <span>d₃:</span>
                          <span className="font-mono">1.638 Å</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {selectedTechnique === 'AFM' && (
              <>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">3D Surface Visualization</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[400px]">
                      {currentImage && (
                        <Surface3DViewer
                          heightMap={currentImage.data}
                          scanSize={afmParams.scanSize as [number, number]}
                        />
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
                          <div className="flex justify-between">
                            <span>Sa (Average):</span>
                            <span className="font-bold">{analysisResults.roughness.Sa.toFixed(2)} nm</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Sq (RMS):</span>
                            <span className="font-bold">{analysisResults.roughness.Sq.toFixed(2)} nm</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Sp (Peak):</span>
                            <span className="font-bold">{analysisResults.roughness.Sp.toFixed(2)} nm</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Sv (Valley):</span>
                            <span className="font-bold">{analysisResults.roughness.Sv.toFixed(2)} nm</span>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <Alert>
                        <Info className="h-4 w-4" />
                        <AlertDescription>
                          No roughness data available
                        </AlertDescription>
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              </>
            )}
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
                    <span>Profile Max</span>
                    <span className="font-mono">8.7 nm</span>
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
                    <dd>DEMO-001</dd>
                    <dt className="text-muted-foreground">Technique:</dt>
                    <dd>{selectedTechnique}</dd>
                    <dt className="text-muted-foreground">Date:</dt>
                    <dd>{new Date().toLocaleDateString()}</dd>
                    <dt className="text-muted-foreground">Operator:</dt>
                    <dd>Lab User</dd>
                  </dl>
                </div>

                <Separator />

                <div>
                  <h3 className="font-semibold mb-2">Key Results</h3>
                  {selectedTechnique === 'SEM' && (
                    <ul className="space-y-1 text-sm">
                      <li>• Particle count: 20</li>
                      <li>• Mean diameter: 35.2 ± 8.4 nm</li>
                      <li>• Circularity: 0.85 ± 0.12</li>
                      <li>• Size uniformity: 76%</li>
                    </ul>
                  )}
                  {selectedTechnique === 'TEM' && (
                    <ul className="space-y-1 text-sm">
                      <li>• Crystal structure: FCC</li>
                      <li>• Lattice parameter: 3.615 Å</li>
                      <li>• Defect density: Low</li>
                      <li>• Sample thickness: ~50 nm</li>
                    </ul>
                  )}
                  {selectedTechnique === 'AFM' && (
                    <ul className="space-y-1 text-sm">
                      <li>• Sa roughness: 2.5 nm</li>
                      <li>• Sq roughness: 3.2 nm</li>
                      <li>• Peak-to-valley: 15.7 nm</li>
                      <li>• Grain size: 45 ± 12 nm</li>
                    </ul>
                  )}
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

// Main Session 10 Component
const Session10MicroscopyInterface: React.FC = () => {
  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Session Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-2xl font-bold">
                Session 10: Microscopy Analysis
              </CardTitle>
              <CardDescription className="mt-2">
                SEM, TEM, and AFM imaging with automated analysis
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Badge>Morphology</Badge>
              <Badge>Structure</Badge>
              <Badge>Surface</Badge>
              <Badge>Composition</Badge>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Main Interface */}
      <MicroscopyInterface />

      {/* Session Footer */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Session Progress</p>
              <Progress value={100} className="w-[200px]" />
            </div>
            <div className="text-sm text-muted-foreground">
              Platform Completion: 62.5% (10/16 sessions)
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Session10MicroscopyInterface;
