import { useCallback, useState } from 'react';
import { ForecastingTool } from './components/ForecastingTool';
import { ModelAnalytics } from './components/ModelAnalytics';
import { GlobalModelAnalytics } from './components/GlobalModelAnalytics';
import { Help } from './components/Help';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Sun, BarChart3, FlaskConical, HelpCircle } from 'lucide-react';

export default function App() {
  const [selectedCoordinates, setSelectedCoordinates] = useState({ lat: 7.0731, lng: 125.6128 });

  const handleCoordinatesChange = useCallback((lat: number, lng: number) => {
    setSelectedCoordinates((previous) => {
      if (previous.lat === lat && previous.lng === lng) {
        return previous;
      }
      return { lat, lng };
    });
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-orange-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-5">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-gradient-to-br from-yellow-400 via-orange-500 to-red-500 rounded-xl flex items-center justify-center shadow-lg">
              <Sun className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-xl sm:text-2xl bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                Solar Energy Potential Forecasting
              </h1>
              <p className="text-xs text-gray-600">FI-AdaBoost Regression Model</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <Tabs defaultValue="forecasting" className="w-full">
          <TabsList className="grid w-full max-w-3xl mx-auto grid-cols-4 mb-8 h-12 p-1 bg-white shadow-md border border-gray-200">
            <TabsTrigger 
              value="forecasting" 
              className="flex items-center gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-orange-500 data-[state=active]:to-amber-500 data-[state=active]:text-white data-[state=active]:shadow-md transition-all"
            >
              <Sun className="w-4 h-4" />
              <span className="hidden sm:inline">Forecasting Tool</span>
              <span className="sm:hidden">Forecast</span>
            </TabsTrigger>
            <TabsTrigger 
              value="analytics" 
              className="flex items-center gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-500 data-[state=active]:to-cyan-600 data-[state=active]:text-white data-[state=active]:shadow-md transition-all"
            >
              <BarChart3 className="w-4 h-4" />
              <span className="hidden sm:inline">Location Analysis</span>
              <span className="sm:hidden">Location</span>
            </TabsTrigger>
            <TabsTrigger
              value="global"
              className="flex items-center gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-violet-500 data-[state=active]:to-fuchsia-600 data-[state=active]:text-white data-[state=active]:shadow-md transition-all"
            >
              <FlaskConical className="w-4 h-4" />
              <span className="hidden sm:inline">Model Analysis</span>
              <span className="sm:hidden">Global</span>
            </TabsTrigger>
            <TabsTrigger
              value="help"
              className="flex items-center gap-2 data-[state=active]:bg-gradient-to-r data-[state=active]:from-slate-600 data-[state=active]:to-slate-700 data-[state=active]:text-white data-[state=active]:shadow-md transition-all"
            >
              <HelpCircle className="w-4 h-4" />
              <span className="hidden sm:inline">Help</span>
              <span className="sm:hidden">Help</span>
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="forecasting" className="mt-0">
            <ForecastingTool
              onCoordinatesChange={handleCoordinatesChange}
            />
          </TabsContent>
          
          <TabsContent value="analytics" className="mt-0">
            <ModelAnalytics lat={selectedCoordinates.lat} lng={selectedCoordinates.lng} />
          </TabsContent>

          <TabsContent value="global" className="mt-0">
            <GlobalModelAnalytics />
          </TabsContent>

          <TabsContent value="help" className="mt-0">
            <Help />
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      {/* <footer className="bg-white/80 backdrop-blur-md border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 py-8 text-center">
          <p className="text-sm text-gray-700">Solar Energy Potential Forecasting - Davao City</p>
          <p className="text-xs text-gray-500 mt-1">Thesis Project • Feature-Importance-Aware AdaBoost Research</p>
          <p className="text-xs text-gray-400 mt-2">© 2024 CCS Research • Demonstrating 18.5% RMSE Improvement</p>
        </div>
      </footer> */}
    </div>
  );
}