import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { MapPin, Locate, Sun, Home, Compass, Cloud, Droplets, Thermometer, Navigation, Search, ChevronDown, ChevronUp, BookOpen } from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { MapContainer, TileLayer, Marker, useMapEvents, useMap } from "react-leaflet";
import L from "leaflet";
import type { LatLngBoundsExpression, LeafletMouseEvent } from "leaflet";
import { fetchBackendJson } from '../lib/backend';

const defaultIcon = L.icon({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});

const DEFAULT_MAP_CENTER: [number, number] = [7.0731, 125.6128];
const DEFAULT_COORDINATES = {
  lat: DEFAULT_MAP_CENTER[0].toFixed(6),
  lng: DEFAULT_MAP_CENTER[1].toFixed(6),
};
const DAVAO_BOUNDS = {
  south: 6.8,
  west: 125.3,
  north: 7.35,
  east: 126.0,
} as const;
const DAVAO_MAP_BOUNDS: LatLngBoundsExpression = [
  [DAVAO_BOUNDS.south, DAVAO_BOUNDS.west],
  [DAVAO_BOUNDS.north, DAVAO_BOUNDS.east],
];
const DAVAO_BOUNDS_ERROR =
  'Please choose a location inside Davao City only. Use the map, search, or coordinates within the Davao City bounds.';

function formatCoordinate(value: number): string {
  return value.toFixed(6);
}

function isWithinDavaoBounds(lat: number, lng: number): boolean {
  return (
    lat >= DAVAO_BOUNDS.south &&
    lat <= DAVAO_BOUNDS.north &&
    lng >= DAVAO_BOUNDS.west &&
    lng <= DAVAO_BOUNDS.east
  );
}

function mapLiveLookupError(message: string): string {
  if (/Failed to fetch|NetworkError|Backend URL is not configured/i.test(message)) {
    return 'Failed to reach backend. Set VITE_BACKEND_URL to your Railway backend URL, then redeploy frontend.';
  }
  if (/No mapped rooftop contains the selected point/i.test(message)) {
    return 'No mapped rooftop contains this point. Zoom in and click inside the building outline.';
  }
  if (/No mapped building footprint was found near/i.test(message)) {
    return 'No mapped rooftop was found near this pinned location. Zoom in and click directly inside a mapped building outline.';
  }
  if (/Live rooftop lookup is temporarily unavailable because all configured Overpass endpoints failed/i.test(message)) {
    return 'Live OpenStreetMap rooftop lookup is temporarily unavailable right now. Please try predicting again shortly.';
  }
  return message;
}

function parseCoordinatePair(input: string): { lat: number; lng: number } | { error: string } | null {
  const trimmed = input.trim();
  const parts = trimmed.split(",");

  if (parts.length !== 2) {
    return null;
  }

  const [latPart, lngPart] = parts.map((part) => part.trim());
  const coordinateToken = /^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$/;
  const latLooksNumeric = /^[+-]?(?:\d|\.\d)/.test(latPart);
  const lngLooksNumeric = /^[+-]?(?:\d|\.\d)/.test(lngPart);

  if (!latLooksNumeric && !lngLooksNumeric) {
    return null;
  }

  if (!coordinateToken.test(latPart) || !coordinateToken.test(lngPart)) {
    return {
      error: 'Enter coordinates as "latitude, longitude" using decimal values, for example 7.073100, 125.612800.',
    };
  }

  const lat = Number(latPart);
  const lng = Number(lngPart);
  if (lat < -90 || lat > 90 || lng < -180 || lng > 180) {
    return {
      error: "Coordinates must use latitude between -90 and 90 and longitude between -180 and 180.",
    };
  }

  if (!isWithinDavaoBounds(lat, lng)) {
    return {
      error: `Coordinates must be inside Davao City (latitude ${DAVAO_BOUNDS.south.toFixed(2)}-${DAVAO_BOUNDS.north.toFixed(2)}, longitude ${DAVAO_BOUNDS.west.toFixed(2)}-${DAVAO_BOUNDS.east.toFixed(2)}).`,
    };
  }

  return { lat, lng };
}


interface PredictionResult {
  solarPotential: number;
  predictedIrradiance: number;
  rooftopArea: number;
  solarExposureIndex: number;
  orientation: string;
  azimuth: number;
  sunshineHours: number;
  cloudCover: number;
  temperature: number;
  humidity: number;
  clearSkyRatio: number;
}

type ApiPredictionResult = Omit<PredictionResult, 'predictedIrradiance'> & {
  predictedIrradiance?: number;
};

function normalizePredictionResult(result: ApiPredictionResult): PredictionResult {
  if (typeof result.predictedIrradiance === 'number') {
    return {
      ...result,
      predictedIrradiance: result.predictedIrradiance,
    };
  }

  const looksLikeLegacyIrradiance = result.solarPotential <= 15;
  const predictedIrradiance = looksLikeLegacyIrradiance
    ? result.solarPotential
    : result.solarPotential / Math.max(result.rooftopArea, 1);

  return {
    ...result,
    predictedIrradiance,
    solarPotential: looksLikeLegacyIrradiance
      ? result.solarPotential * result.rooftopArea
      : result.solarPotential,
  };
}

function ClickToSetLocation({ onPick }: { onPick: (lat: number, lng: number) => void }) {
  useMapEvents({
    click(e: LeafletMouseEvent) {
      onPick(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

function RecenterMap({ lat, lng, zoom = 15 }: { lat: number; lng: number; zoom?: number }) {
  const map = useMap();

  useEffect(() => {
    if (!Number.isFinite(lat) || !Number.isFinite(lng)) return;
    map.setView([lat, lng], map.getZoom() || zoom, { animate: true });
  }, [lat, lng, zoom, map]);

  return null;
}


interface ForecastingToolProps {
  onCoordinatesChange?: (lat: number, lng: number) => void;
}

export function ForecastingTool({ onCoordinatesChange }: ForecastingToolProps) {
  const [address, setAddress] = useState('Davao City, Philippines');
  const [coordinates, setCoordinates] = useState(DEFAULT_COORDINATES);
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [showGuide, setShowGuide] = useState(true);
  const latValue = Number(coordinates.lat);
  const lngValue = Number(coordinates.lng);
  const hasValidCoordinates = Number.isFinite(latValue) && Number.isFinite(lngValue);
  const isWithinDavao = hasValidCoordinates && isWithinDavaoBounds(latValue, lngValue);
  const canUseCoordinates = hasValidCoordinates && isWithinDavao;

  const annualEnergyPotential = prediction
    ? prediction.solarPotential * 365
    : null;

  // Clear prediction when coordinates change
  useEffect(() => {
    if (coordinates.lat || coordinates.lng) {
      setPrediction(null);
    }
  }, [coordinates.lat, coordinates.lng]);

  useEffect(() => {
    if (canUseCoordinates) {
      onCoordinatesChange?.(latValue, lngValue);
    }
  }, [canUseCoordinates, latValue, lngValue, onCoordinatesChange]);

  const applyCoordinates = (lat: number, lng: number, nextAddress?: string): boolean => {
    if (!isWithinDavaoBounds(lat, lng)) {
      setApiError(DAVAO_BOUNDS_ERROR);
      return false;
    }

    const formattedLat = formatCoordinate(lat);
    const formattedLng = formatCoordinate(lng);
    setCoordinates({ lat: formattedLat, lng: formattedLng });
    setAddress(nextAddress ?? `${formattedLat}, ${formattedLng}`);
    setApiError(null);
    return true;
  };

  const handleLocateMe = () => {
    setIsLoading(true);
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          if (!isWithinDavaoBounds(position.coords.latitude, position.coords.longitude)) {
            applyCoordinates(DEFAULT_MAP_CENTER[0], DEFAULT_MAP_CENTER[1], 'Davao City, Philippines');
            setApiError('Your current location is outside Davao City, so the tool stayed at the Davao City center.');
            setIsLoading(false);
            return;
          }

          applyCoordinates(position.coords.latitude, position.coords.longitude);
          setIsLoading(false);
        },
        () => {
          applyCoordinates(DEFAULT_MAP_CENTER[0], DEFAULT_MAP_CENTER[1], 'Davao City, Philippines');
          setIsLoading(false);
        }
      );
    } else {
      applyCoordinates(DEFAULT_MAP_CENTER[0], DEFAULT_MAP_CENTER[1], 'Davao City, Philippines');
      setIsLoading(false);
    }
  };

  const handleSearchAddress = async () => {
    const trimmedAddress = address.trim();
    if (!trimmedAddress) {
      setApiError('Please enter an address or coordinates to search.');
      return;
    }

    const parsedCoordinates = parseCoordinatePair(trimmedAddress);
    if (parsedCoordinates && "error" in parsedCoordinates) {
      setApiError(parsedCoordinates.error);
      return;
    }

    if (parsedCoordinates) {
      applyCoordinates(parsedCoordinates.lat, parsedCoordinates.lng);
      return;
    }

    setIsLoading(true);
    setApiError(null);
    try {
      const query = encodeURIComponent(`${trimmedAddress}, Davao City, Philippines`);
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${query}&limit=1&bounded=1&viewbox=${DAVAO_BOUNDS.west},${DAVAO_BOUNDS.south},${DAVAO_BOUNDS.east},${DAVAO_BOUNDS.north}`,
        {
          headers: {
            'User-Agent': 'SolarEnergyForecastingApp/1.0'
          }
        }
      );

      if (!response.ok) {
        throw new Error('Geocoding service unavailable');
      }

      const results = await response.json();

      if (results.length === 0) {
        setApiError(`Address not found: "${address}". Try including street name or landmark.`);
        setIsLoading(false);
        return;
      }

      const location = results[0];
      const lat = Number(location.lat);
      const lng = Number(location.lon);
      if (!isWithinDavaoBounds(lat, lng)) {
        setApiError('Address result is outside Davao City. Try a Davao City landmark, barangay, or decimal coordinate.');
        setIsLoading(false);
        return;
      }

      applyCoordinates(lat, lng, location.display_name);
      console.log('📍 Geocoded address:', location.display_name, '→', formatCoordinate(lat), formatCoordinate(lng));

    } catch (error) {
      console.error('Geocoding error:', error);
      setApiError('Unable to find address. Please try entering coordinates manually or clicking on the map.');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!hasValidCoordinates) {
      setApiError('Please provide valid latitude and longitude values.');
      return;
    }
    if (!isWithinDavao) {
      setApiError(DAVAO_BOUNDS_ERROR);
      return;
    }

    console.log('🎯 Predicting for coordinates:', { lat: latValue, lng: lngValue });

    setApiError(null);
    setIsLoading(true);

    try {
      const result = await fetchBackendJson<ApiPredictionResult>('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ lat: latValue, lng: lngValue }),
      });
      console.log('✅ Prediction result:', result);
      setPrediction(normalizePredictionResult(result));
    } catch (error) {
      console.error('❌ Prediction error:', error);
      const message = error instanceof Error ? error.message : 'Unable to reach prediction service.';
      setApiError(mapLiveLookupError(message));
      setPrediction(null);
    } finally {
      setIsLoading(false);
    }
  };

  const getSolarRating = (irradiance: number): { label: string; color: string; description: string; bgGradient: string } => {
    if (irradiance >= 5.5) return { 
      label: 'Excellent', 
      color: 'bg-emerald-500',
      bgGradient: 'from-emerald-500 to-green-600',
      description: 'Outstanding rooftop solar energy potential for the available roof area.'
    };
    if (irradiance >= 4.5) return { 
      label: 'Very Good', 
      color: 'bg-blue-500',
      bgGradient: 'from-blue-500 to-cyan-600',
      description: 'Strong rooftop solar energy potential for the selected building footprint.'
    };
    if (irradiance >= 3.5) return { 
      label: 'Good', 
      color: 'bg-amber-500',
      bgGradient: 'from-amber-500 to-yellow-600',
      description: 'Moderate rooftop solar energy potential based on irradiance and mapped area.'
    };
    return { 
      label: 'Fair', 
      color: 'bg-orange-500',
      bgGradient: 'from-orange-500 to-red-600',
      description: 'Lower rooftop solar energy potential compared with stronger irradiance locations.'
    };
  };

  const rating = prediction ? getSolarRating(prediction.predictedIrradiance) : null;

  return (
    <div className="space-y-6">
      {/* Input Section */}
      <Card className="border-2 border-blue-100 shadow-lg">
        <CardHeader className="bg-gradient-to-r from-blue-50 to-cyan-50">
          <CardTitle className="flex items-center gap-2">
            <MapPin className="w-5 h-5 text-blue-600" />
            Select Your Location
          </CardTitle>
          <CardDescription>
            Enter an address, paste coordinates, or click directly inside a mapped rooftop footprint in Davao City
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-6 space-y-4">
          {/* Address Input */}
          <div className="space-y-2">
            <Label htmlFor="address" className="text-sm">Address or "latitude, longitude"</Label>
            <div className="flex gap-2">
              <Input
                id="address"
                placeholder='e.g., SM City Davao or 7.073100, 125.612800'
                value={address}
                onChange={(e) => setAddress(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    handleSearchAddress();
                  }
                }}
                className="flex-1"
              />
              <Button
                variant="default"
                size="icon"
                onClick={handleSearchAddress}
                disabled={isLoading || !address.trim()}
                title="Search this address or use the typed coordinates"
                className="shrink-0 bg-blue-600 hover:bg-blue-700"
              >
                <Search className="w-4 h-4" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={handleLocateMe}
                disabled={isLoading}
                title="Use my current location"
                className="shrink-0"
              >
                <Locate className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {/* Coordinates Input */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="latitude" className="text-sm">Latitude</Label>
              <Input
                id="latitude"
                placeholder="e.g., 7.0731"
                value={coordinates.lat}
                inputMode="decimal"
                onChange={(e) => {
                  setCoordinates({ ...coordinates, lat: e.target.value });
                  setApiError(null);
                }}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="longitude" className="text-sm">Longitude</Label>
              <Input
                id="longitude"
                placeholder="e.g., 125.6128"
                value={coordinates.lng}
                inputMode="decimal"
                onChange={(e) => {
                  setCoordinates({ ...coordinates, lng: e.target.value });
                  setApiError(null);
                }}
              />
            </div>
          </div>

<div className="space-y-2">
  <Label className="text-sm">Pin Your Exact Location on Map</Label>

 {/* Map Picker */}
<div className="rounded-lg overflow-hidden border-2 border-blue-300 shadow-lg h-64 sm:h-96 md:h-[500px] relative z-0">
  <MapContainer
    center={DEFAULT_MAP_CENTER}
    zoom={14}
    minZoom={12}
    maxZoom={18}
    maxBounds={DAVAO_MAP_BOUNDS}
    maxBoundsViscosity={1.0}
    style={{ height: "100%", width: "100%", background: "#e5e7eb" }}
    scrollWheelZoom
  >
    <TileLayer
      attribution='&copy; OpenStreetMap contributors'
      url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
    />

    <ClickToSetLocation
      onPick={(lat, lng) => {
        applyCoordinates(lat, lng);
      }}
    />

    {canUseCoordinates && (
      <>
        <RecenterMap lat={latValue} lng={lngValue} />
        <Marker
          position={[latValue, lngValue]}
          icon={defaultIcon}
        />
      </>
    )}
  </MapContainer>
</div>


  <p className="text-xs text-gray-500 flex items-center gap-1">
    <Navigation className="w-3 h-3" />
    Click directly inside a mapped rooftop footprint within the locked Davao City bounds
  </p>
</div>


          {coordinates.lat && coordinates.lng && (
            <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
              <p className="text-sm">
                <span className="text-gray-600">Selected coordinates:</span>{' '}
                <span className="font-mono">{coordinates.lat}, {coordinates.lng}</span>
              </p>
              {hasValidCoordinates && !isWithinDavao && (
                <p className="text-xs text-red-600 mt-1 font-medium">
                  Location must be inside Davao City bounds before prediction.
                </p>
              )}
              {!prediction && canUseCoordinates && (
                <p className="text-xs text-blue-600 mt-1 font-medium">
                  📍 New location selected - Click "Predict Solar Potential" to analyze
                </p>
              )}
            </div>
          )}

          <Button
            className="w-full bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 text-white shadow-lg"
            onClick={handlePredict}
            disabled={isLoading || !canUseCoordinates}
            size="lg"
          >
            {isLoading ? (
              <>
                <div className="w-4 h-4 mr-2 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Analyzing Location...
              </>
            ) : (
              <>
                <Sun className="w-5 h-5 mr-2" />
                Predict Solar Potential
              </>
            )}
          </Button>

          {apiError && (
            <div className="p-3 bg-red-50 rounded-lg border border-red-200">
              <p className="text-sm text-red-700">{apiError}</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* How to Use Guide */}
      <Card className="border-2 border-amber-200 bg-amber-50 shadow-md">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-base">
              <BookOpen className="w-4 h-4 text-amber-600" />
              How to Use This Tool
            </CardTitle>
            <button
              onClick={() => setShowGuide((v) => !v)}
              className="flex items-center gap-1 text-xs text-amber-700 hover:text-amber-900 font-medium"
            >
              {showGuide ? <><ChevronUp className="w-4 h-4" /> Hide guide</> : <><ChevronDown className="w-4 h-4" /> Show guide</>}
            </button>
          </div>
        </CardHeader>
        {showGuide && (
          <CardContent className="space-y-5 text-sm text-gray-700 pt-0">
            {/* A. Prototype Overview */}
            <div>
              <p className="font-semibold text-gray-800 mb-1">A. Prototype Overview</p>
              <div className="space-y-2 text-xs leading-relaxed text-gray-600">
                <p>
                  Solar Energy Potential Forecasting estimates the theoretical sunlight energy available on rooftops in Davao City using FI-AdaBoost. It does not compute actual PV electricity output because panel efficiency, inverter losses, panel layout, and installation constraints are outside the study scope.
                </p>
                <p>
                  Live inputs come from OpenStreetMap/Overpass for rooftop geometry, NASA POWER for solar and weather context, and pvlib for clear-sky and sunshine-hour estimates.
                </p>
              </div>
            </div>

            {/* B. Pinning a Location */}
            <div>
              <p className="font-semibold text-gray-800 mb-1">B. Pinning a Location</p>
              <ul className="space-y-1 list-disc list-inside text-xs text-gray-600">
                <li>Search an address — results are filtered to Davao City bounds.</li>
                <li>Type decimal coordinates inside Davao City in <span className="font-mono bg-amber-100 px-1 rounded">lat, lng</span> format (e.g. <span className="font-mono bg-amber-100 px-1 rounded">7.0731, 125.6128</span>) and press Enter.</li>
                <li>Click directly inside a mapped rooftop footprint to drop the pin at that exact point.</li>
              </ul>
            </div>

            {/* C. What the Prediction Shows */}
            <div>
              <p className="font-semibold text-gray-800 mb-2">C. What the Prediction Shows</p>
              <div className="space-y-2 text-xs">
                {[
                  ['Solar Energy Potential (SEP)', 'Predicted irradiance (kWh/m²/day) × detected rooftop area (m²) = theoretical rooftop solar energy in kWh/day.'],
                  ['Irradiance Resource Level', 'Predicted GHI ÷ 7 kWh/m²/day × 100. Ratings use the irradiance thresholds: Excellent ≥ 5.5, Very Good ≥ 4.5, Good ≥ 3.5, Fair < 3.5 kWh/m²/day.'],
                  ['Solar Exposure Index (SEI)', 'Normalized exposure score from rooftop area × orientation score × (1 − shading factor) × tilt factor. Used directly as a training feature.'],
                  ['Azimuth', 'Compass bearing of the building\'s longest edge. 0° = North, 90° = East, 180° = South, 270° = West.'],
                  ['Clear-Sky Ratio', 'Actual GHI ÷ pvlib Ineichen theoretical clear-sky GHI. Values near 1.0 indicate mostly clear conditions year-round.'],
                ].map(([term, def]) => (
                  <div key={term} className="grid grid-cols-[180px_1fr] gap-2 rounded-lg border border-amber-200 bg-white p-2">
                    <span className="font-medium text-gray-800">{term}</span>
                    <span className="text-gray-600">{def}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* D. Scope & Assumptions */}
            <div className="rounded-xl border border-amber-300 bg-amber-100 p-3 text-xs text-amber-900 space-y-1">
              <p className="font-semibold uppercase tracking-wide">Scope &amp; Assumptions</p>
              <ul className="list-disc list-inside space-y-0.5">
                <li>Whole rooftop area is assumed available — no panel layout is modelled.</li>
                <li>Panel efficiency, inverter losses, and performance ratio are excluded.</li>
                <li>Prediction only runs when the selected point is inside an OpenStreetMap building polygon; accuracy depends on OSM coverage quality.</li>
                <li>Locations outside Davao City bounds are blocked by the map, search, coordinate, and prediction controls.</li>
              </ul>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Results Section */}
      {prediction && (
        <div className="space-y-6">
          {/* Location Info Banner */}
          <div className="p-4 bg-gradient-to-r from-blue-50 to-cyan-50 rounded-lg border-2 border-blue-200 shadow-sm">
            <div className="flex items-center gap-2 mb-2">
              <MapPin className="w-4 h-4 text-blue-600" />
              <span className="text-sm font-semibold text-blue-900">Analysis for Location:</span>
            </div>
            <p className="text-sm text-gray-700">
              <span className="font-mono bg-white px-2 py-1 rounded border border-blue-200">
                {coordinates.lat}, {coordinates.lng}
              </span>
            </p>
            {address && address !== `${coordinates.lat}, ${coordinates.lng}` && (
              <p className="text-xs text-gray-600 mt-1">{address}</p>
            )}
          </div>

          {/* Solar Potential Card */}
          <Card className={`border-2 shadow-2xl bg-gradient-to-br ${rating?.color === 'bg-emerald-500' ? 'from-emerald-50 via-green-50 to-teal-50 border-emerald-300' : rating?.color === 'bg-blue-500' ? 'from-blue-50 via-cyan-50 to-sky-50 border-blue-300' : rating?.color === 'bg-amber-500' ? 'from-amber-50 via-yellow-50 to-orange-50 border-amber-300' : 'from-orange-50 via-red-50 to-rose-50 border-orange-300'}`}>
            <CardHeader>
              <div className="flex items-start justify-between">
                <div>
                  <CardTitle className="text-2xl">Predicted Solar Energy Potential</CardTitle>
                  <CardDescription className="text-sm">
                    FI-AdaBoost first predicts solar irradiance, then the app multiplies it by the OpenStreetMap rooftop area to show theoretical solar energy potential.
                  </CardDescription>
                </div>
                <div className={`w-16 h-16 bg-gradient-to-br ${rating?.bgGradient} rounded-full flex items-center justify-center shadow-lg`}>
                  <Sun className="w-8 h-8 text-white" />
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div>
                  <p className="text-xs text-gray-500 uppercase tracking-wide mb-2">Theoretical Solar Energy Potential</p>
                  <div className="flex items-baseline gap-3 mb-3">
                    <span className="text-6xl sm:text-7xl bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                      {prediction.solarPotential.toFixed(2)}
                    </span>
                    <span className="text-2xl text-gray-600">kWh/day</span>
                  </div>
                  <Badge className={`${rating?.color} text-white shadow-md px-4 py-2 text-sm`}>
                    {rating?.label} Rating
                  </Badge>
                </div>

                {/* Visual Indicator */}
                <div className="space-y-3">
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-gray-700">Irradiance Resource Level</span>
                    <span className="text-lg">{Math.round((prediction.predictedIrradiance / 7) * 100)}%</span>
                  </div>
                  <div className="h-4 bg-gray-200/50 rounded-full overflow-hidden shadow-inner">
                    <div
                      className={`h-full bg-gradient-to-r ${rating?.bgGradient} transition-all duration-1000 shadow-lg`}
                      style={{ width: `${Math.min((prediction.predictedIrradiance / 7) * 100, 100)}%` }}
                    />
                  </div>
                </div>

                <div className="bg-white/70 backdrop-blur rounded-xl p-4 shadow-md border border-white/50">
                  <p className="text-sm text-gray-800 leading-relaxed">
                    <strong className="text-gray-900">What this means:</strong> {rating?.description}
                  </p>
                </div>

                {annualEnergyPotential !== null && (
                  <div className="rounded-xl border border-amber-200 bg-white p-4 shadow-sm">
                    <p className="text-xs text-slate-500 uppercase tracking-wide">Calculation Used</p>
                    <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-3">
                      <p className="text-sm text-slate-700">
                        Predicted irradiance: <span className="font-semibold">{prediction.predictedIrradiance.toFixed(2)} kWh/m²/day</span>
                      </p>
                      <p className="text-sm text-slate-700">
                        Rooftop area: <span className="font-semibold">{prediction.rooftopArea.toFixed(1)} m²</span>
                      </p>
                      <p className="text-sm text-slate-700">
                        Annual potential: <span className="font-semibold">{annualEnergyPotential.toLocaleString(undefined, { maximumFractionDigits: 0 })} kWh/year</span>
                      </p>
                    </div>
                    <p className="mt-3 text-sm text-slate-700">
                      Daily SEP formula: <span className="font-semibold">{prediction.predictedIrradiance.toFixed(2)} kWh/m²/day × {prediction.rooftopArea.toFixed(1)} m² = {prediction.solarPotential.toFixed(2)} kWh/day</span>
                    </p>
                    <p className="text-xs text-slate-400 mt-3 leading-relaxed">
                      This is theoretical rooftop energy before any conversion to electricity; panel efficiency and performance ratio are outside this study scope.
                    </p>
                    <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-3">
                      <p className="text-xs font-semibold uppercase tracking-wide text-amber-900">Scope and Assumptions</p>
                      <p className="mt-2 text-xs leading-relaxed text-amber-900">
                        This result assumes the whole mapped rooftop area is available for solar collection. It represents incident sunlight on the roof, not usable PV electricity, and excludes specific panel layout, sensor data, panel efficiency, inverter losses, and performance ratio.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Rooftop Information */}
          <Card className="border-2 border-purple-100 shadow-lg">
            <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50">
              <CardTitle className="flex items-center gap-2">
                <Home className="w-5 h-5 text-purple-600" />
                Key Rooftop Information
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <InfoItem
                  label="Rooftop Area"
                  value={`${prediction.rooftopArea.toFixed(1)} m²`}
                  description="Total rooftop surface area"
                  gradient="from-purple-500 to-pink-500"
                />
                <InfoItem
                  label="Solar Exposure Index (SEI)"
                  value={prediction.solarExposureIndex.toFixed(3)}
                  description="Solar radiation exposure level"
                  gradient="from-orange-500 to-red-500"
                />
                <InfoItem
                  label="Orientation"
                  value={prediction.orientation}
                  icon={<Compass className="w-5 h-5 text-blue-500" />}
                  description="Primary roof facing direction"
                  gradient="from-blue-500 to-cyan-500"
                />
                <InfoItem
                  label="Azimuth Angle"
                  value={`${prediction.azimuth.toFixed(1)}°`}
                  description="Optimal panel angle"
                  gradient="from-green-500 to-emerald-500"
                />
              </div>
            </CardContent>
          </Card>

          {/* Regional Weather Context */}
          <Card className="border-2 border-cyan-100 shadow-lg">
            <CardHeader className="bg-gradient-to-r from-cyan-50 to-blue-50">
              <CardTitle className="flex items-center gap-2">
                <Cloud className="w-5 h-5 text-cyan-600" />
                Regional Weather Context
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <InfoItem
                  label="Sunshine Hours"
                  value={`${prediction.sunshineHours.toFixed(1)} hrs/day`}
                  icon={<Sun className="w-5 h-5 text-yellow-500" />}
                  description="Average daily sunshine"
                  gradient="from-yellow-500 to-orange-500"
                />
                <InfoItem
                  label="Cloud Cover"
                  value={`${prediction.cloudCover.toFixed(1)}%`}
                  icon={<Cloud className="w-5 h-5 text-gray-500" />}
                  description="Average cloud coverage"
                  gradient="from-gray-500 to-slate-500"
                />
                <InfoItem
                  label="Temperature"
                  value={`${prediction.temperature.toFixed(1)}°C`}
                  icon={<Thermometer className="w-5 h-5 text-red-500" />}
                  description="Average temperature"
                  gradient="from-red-500 to-orange-500"
                />
                <InfoItem
                  label="Humidity"
                  value={`${prediction.humidity.toFixed(1)}%`}
                  icon={<Droplets className="w-5 h-5 text-blue-500" />}
                  description="Relative humidity"
                  gradient="from-blue-500 to-indigo-500"
                />
              </div>

            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

interface InfoItemProps {
  label: string;
  value: string;
  icon?: ReactNode;
  description?: string;
  gradient: string;
}

function InfoItem({ label, value, icon, description, gradient }: InfoItemProps) {
  return (
    <div className="space-y-2 p-4 rounded-xl bg-white shadow-md border border-gray-100 hover:shadow-lg transition-shadow">
      <div className="flex items-center gap-2">
        {icon}
        <span className="text-sm text-gray-600">{label}</span>
      </div>
      <div className={`text-3xl bg-gradient-to-r ${gradient} bg-clip-text text-transparent`}>
        {value}
      </div>
      {description && <p className="text-xs text-gray-500 leading-relaxed">{description}</p>}
    </div>
  );
}
