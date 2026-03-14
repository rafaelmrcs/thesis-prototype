import { useState } from 'react';
import { Building2, MapPin, Sun, Maximize2, Compass, TrendingUp, Search } from 'lucide-react';

// Sample rooftop data
const generateRooftops = () => {
  const barangays = ['Poblacion', 'Buhangin', 'Paquibato', 'Talomo', 'Toril', 'Agdao', 'Matina'];
  const orientations = ['North', 'South', 'East', 'West', 'Northeast', 'Southeast', 'Northwest', 'Southwest'];
  
  return Array.from({ length: 50 }, (_, i) => ({
    id: `RT-${String(i + 1).padStart(4, '0')}`,
    barangay: barangays[Math.floor(Math.random() * barangays.length)],
    area: parseFloat((80 + Math.random() * 220).toFixed(1)),
    orientation: orientations[Math.floor(Math.random() * orientations.length)],
    solarExposure: parseFloat((0.5 + Math.random() * 0.5).toFixed(3)),
    predictedGHI: parseFloat((4 + Math.random() * 3).toFixed(2)),
    lat: 7.07 + Math.random() * 0.3,
    lon: 125.4 + Math.random() * 0.3,
  }));
};

const rooftops = generateRooftops();

export function RooftopExplorer() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedRooftop, setSelectedRooftop] = useState(rooftops[0]);
  const [sortBy, setSortBy] = useState<'area' | 'solarExposure' | 'predictedGHI'>('solarExposure');

  const filteredRooftops = rooftops
    .filter(r => 
      r.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      r.barangay.toLowerCase().includes(searchTerm.toLowerCase())
    )
    .sort((a, b) => b[sortBy] - a[sortBy]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl p-6 shadow-md">
        <h2 className="text-xl mb-2 flex items-center gap-2">
          <Building2 className="w-6 h-6 text-orange-500" />
          Rooftop Solar Potential Explorer
        </h2>
        <p className="text-sm text-gray-600">
          Analyze solar energy potential for individual rooftops in Davao City
        </p>
      </div>

      {/* Search and Filter */}
      <div className="bg-white rounded-xl p-4 shadow-md">
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search by ID or barangay..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
            />
          </div>
          
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
          >
            <option value="solarExposure">Sort by Solar Exposure</option>
            <option value="area">Sort by Area</option>
            <option value="predictedGHI">Sort by Predicted GHI</option>
          </select>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Rooftop List */}
        <div className="lg:col-span-1 space-y-3 max-h-[600px] overflow-y-auto">
          {filteredRooftops.map(rooftop => (
            <div
              key={rooftop.id}
              onClick={() => setSelectedRooftop(rooftop)}
              className={`bg-white rounded-lg p-4 cursor-pointer transition-all ${
                selectedRooftop.id === rooftop.id
                  ? 'ring-2 ring-orange-500 shadow-lg'
                  : 'hover:shadow-md border border-gray-200'
              }`}
            >
              <div className="flex items-start justify-between mb-2">
                <div>
                  <h4 className="text-sm">{rooftop.id}</h4>
                  <p className="text-xs text-gray-600">{rooftop.barangay}</p>
                </div>
                <div className={`px-2 py-1 rounded text-xs ${
                  rooftop.solarExposure > 0.8 ? 'bg-green-100 text-green-700' :
                  rooftop.solarExposure > 0.6 ? 'bg-yellow-100 text-yellow-700' :
                  'bg-orange-100 text-orange-700'
                }`}>
                  SEI: {rooftop.solarExposure.toFixed(2)}
                </div>
              </div>
              
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-gray-600">Area:</span>
                  <span className="ml-1">{rooftop.area} m²</span>
                </div>
                <div>
                  <span className="text-gray-600">GHI:</span>
                  <span className="ml-1">{rooftop.predictedGHI} kWh/m²</span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Selected Rooftop Details */}
        <div className="lg:col-span-2 space-y-4">
          {/* Map Placeholder */}
          <div className="bg-gradient-to-br from-blue-100 to-green-100 rounded-xl p-8 shadow-md relative overflow-hidden h-64">
            <div className="absolute inset-0 opacity-10">
              <svg className="w-full h-full" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="40" fill="none" stroke="currentColor" strokeWidth="0.5" />
                <circle cx="50" cy="50" r="30" fill="none" stroke="currentColor" strokeWidth="0.5" />
                <circle cx="50" cy="50" r="20" fill="none" stroke="currentColor" strokeWidth="0.5" />
                <circle cx="50" cy="50" r="10" fill="none" stroke="currentColor" strokeWidth="0.5" />
              </svg>
            </div>
            
            <div className="relative z-10 text-center">
              <MapPin className="w-12 h-12 text-orange-600 mx-auto mb-3 animate-bounce" />
              <h3 className="text-xl text-gray-800 mb-2">{selectedRooftop.id}</h3>
              <p className="text-sm text-gray-700">{selectedRooftop.barangay}, Davao City</p>
              <p className="text-xs text-gray-600 mt-2">
                {selectedRooftop.lat.toFixed(4)}°N, {selectedRooftop.lon.toFixed(4)}°E
              </p>
            </div>
          </div>

          {/* Rooftop Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="bg-white rounded-lg p-4 shadow-md text-center">
              <Maximize2 className="w-6 h-6 text-blue-500 mx-auto mb-2" />
              <p className="text-xs text-gray-600 mb-1">Rooftop Area</p>
              <p className="text-lg">{selectedRooftop.area}</p>
              <p className="text-xs text-gray-500">m²</p>
            </div>

            <div className="bg-white rounded-lg p-4 shadow-md text-center">
              <Compass className="w-6 h-6 text-green-500 mx-auto mb-2" />
              <p className="text-xs text-gray-600 mb-1">Orientation</p>
              <p className="text-lg">{selectedRooftop.orientation}</p>
              <p className="text-xs text-gray-500">direction</p>
            </div>

            <div className="bg-white rounded-lg p-4 shadow-md text-center">
              <TrendingUp className="w-6 h-6 text-orange-500 mx-auto mb-2" />
              <p className="text-xs text-gray-600 mb-1">Solar Exposure</p>
              <p className="text-lg">{selectedRooftop.solarExposure.toFixed(3)}</p>
              <p className="text-xs text-gray-500">SEI score</p>
            </div>

            <div className="bg-white rounded-lg p-4 shadow-md text-center">
              <Sun className="w-6 h-6 text-yellow-500 mx-auto mb-2" />
              <p className="text-xs text-gray-600 mb-1">Predicted GHI</p>
              <p className="text-lg">{selectedRooftop.predictedGHI}</p>
              <p className="text-xs text-gray-500">kWh/m²/day</p>
            </div>
          </div>

          {/* Detailed Analysis */}
          <div className="bg-white rounded-xl p-6 shadow-md">
            <h3 className="text-lg mb-4">Solar Potential Analysis</h3>
            
            <div className="space-y-4">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-700">Solar Exposure Index (SEI)</span>
                  <span className="text-sm">{selectedRooftop.solarExposure.toFixed(3)}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-orange-400 to-yellow-500 h-2 rounded-full"
                    style={{ width: `${selectedRooftop.solarExposure * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-gray-700">Predicted Daily Energy</span>
                  <span className="text-sm">{selectedRooftop.predictedGHI.toFixed(2)} kWh/m²</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-blue-400 to-green-500 h-2 rounded-full"
                    style={{ width: `${(selectedRooftop.predictedGHI / 7) * 100}%` }}
                  />
                </div>
              </div>

              <div className="pt-4 border-t border-gray-200">
                <h4 className="text-sm mb-3">Estimated Annual Output</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-blue-50 rounded-lg p-3">
                    <p className="text-xs text-blue-600 mb-1">Total Energy</p>
                    <p className="text-xl text-blue-700">
                      {(selectedRooftop.predictedGHI * selectedRooftop.area * 365 / 1000).toFixed(1)}
                    </p>
                    <p className="text-xs text-blue-600">MWh/year</p>
                  </div>
                  
                  <div className="bg-green-50 rounded-lg p-3">
                    <p className="text-xs text-green-600 mb-1">CO₂ Offset</p>
                    <p className="text-xl text-green-700">
                      {(selectedRooftop.predictedGHI * selectedRooftop.area * 365 * 0.7 / 1000).toFixed(1)}
                    </p>
                    <p className="text-xs text-green-600">tons/year</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
