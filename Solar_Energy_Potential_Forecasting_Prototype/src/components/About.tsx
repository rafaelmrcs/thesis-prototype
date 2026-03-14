import { BookOpen, Users, Target, Lightbulb, Award, GitBranch } from 'lucide-react';

export function About() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-6 text-white shadow-xl">
        <h2 className="text-2xl mb-2">About This Project</h2>
        <p className="text-blue-100 text-sm">
          An Optimized AdaBoost Regression Algorithm with Feature-Aware Weighting for Solar Energy Potential Forecasting
        </p>
      </div>

      {/* Research Overview */}
      <div className="bg-white rounded-xl p-6 shadow-md">
        <h3 className="text-lg mb-4 flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-blue-500" />
          Research Overview
        </h3>
        <div className="space-y-3 text-sm text-gray-700">
          <p>
            This thesis project develops a <strong>Feature-Importance-Aware AdaBoost (FI-AdaBoost) Regression</strong> model 
            to forecast solar energy potential on rooftops in Davao City, Philippines.
          </p>
          <p>
            The algorithm optimizes the standard AdaBoost Regression framework by integrating feature importance awareness 
            directly into the weight update mechanism, enabling the model to focus on the most influential predictors 
            such as solar irradiance, cloud cover, and topographical characteristics.
          </p>
          <p>
            This approach improves forecasting accuracy, reduces sensitivity to outliers, and enhances model interpretability—
            critical factors for building public trust in rooftop solar systems.
          </p>
        </div>
      </div>

      {/* Objectives */}
      <div className="bg-white rounded-xl p-6 shadow-md">
        <h3 className="text-lg mb-4 flex items-center gap-2">
          <Target className="w-5 h-5 text-orange-500" />
          Research Objectives
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center flex-shrink-0 text-sm mt-0.5">1</div>
              <p className="text-sm text-gray-700">
                Preprocess statistics on solar irradiance and climatic conditions from NASA POWER and OpenStreetMap
              </p>
            </div>
          </div>

          <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <div className="w-6 h-6 bg-orange-500 text-white rounded-full flex items-center justify-center flex-shrink-0 text-sm mt-0.5">2</div>
              <p className="text-sm text-gray-700">
                Optimize AdaBoost Regression using a feature-importance-aware weighting mechanism
              </p>
            </div>
          </div>

          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <div className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center flex-shrink-0 text-sm mt-0.5">3</div>
              <p className="text-sm text-gray-700">
                Compare and evaluate baseline AdaBoost vs FI-AdaBoost using RMSE, MAE, and R²
              </p>
            </div>
          </div>

          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <div className="w-6 h-6 bg-purple-500 text-white rounded-full flex items-center justify-center flex-shrink-0 text-sm mt-0.5">4</div>
              <p className="text-sm text-gray-700">
                Assess the applicability of FI-AdaBoost for data-driven decision-making in solar energy
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Key Innovation */}
      <div className="bg-gradient-to-br from-yellow-50 to-orange-50 border border-orange-200 rounded-xl p-6 shadow-md">
        <h3 className="text-lg mb-4 flex items-center gap-2">
          <Lightbulb className="w-5 h-5 text-orange-600" />
          Key Innovation
        </h3>
        <div className="space-y-3">
          <p className="text-sm text-gray-700">
            Unlike standard AdaBoost Regression, which updates sample weights based solely on prediction error, 
            <strong> FI-AdaBoost incorporates feature importance</strong> into each boosting iteration.
          </p>
          
          <div className="bg-white rounded-lg p-4 border border-orange-200">
            <p className="text-sm mb-2"><strong>Optimized Weight Update:</strong></p>
            <code className="text-xs bg-gray-100 p-2 rounded block overflow-x-auto">
              w<sub>i</sub><sup>(t+1)</sup> = w<sub>i</sub><sup>(t)</sup> × β<sub>t</sub><sup>(1 - e<sub>i</sub><sup>(t)</sup> × Φ(x<sub>i</sub>))</sup> / Z<sub>t</sub>
            </code>
            <p className="text-xs text-gray-600 mt-2">
              Where Φ(x<sub>i</sub>) is the composite feature importance weight for sample x<sub>i</sub>
            </p>
          </div>

          <p className="text-sm text-gray-700">
            This enables the model to apply stronger error corrections for samples involving high-importance features 
            (e.g., Clear Sky Ratio, Sunshine Hours, Solar Exposure Index), leading to improved accuracy and robustness.
          </p>
        </div>
      </div>

      {/* Results */}
      <div className="bg-white rounded-xl p-6 shadow-md">
        <h3 className="text-lg mb-4 flex items-center gap-2">
          <Award className="w-5 h-5 text-green-500" />
          Performance Results
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
          <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-center">
            <p className="text-sm text-green-700 mb-1">R² Improvement</p>
            <p className="text-3xl text-green-800">+2.1%</p>
            <p className="text-xs text-green-600">0.9659 → 0.9867</p>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-center">
            <p className="text-sm text-blue-700 mb-1">RMSE Reduction</p>
            <p className="text-3xl text-blue-800">-18.5%</p>
            <p className="text-xs text-blue-600">0.223 → 0.182 kWh/m²/day</p>
          </div>

          <div className="bg-orange-50 border border-orange-200 rounded-lg p-4 text-center">
            <p className="text-sm text-orange-700 mb-1">MAE Reduction</p>
            <p className="text-3xl text-orange-800">-15.2%</p>
            <p className="text-xs text-orange-600">0.158 → 0.134 kWh/m²/day</p>
          </div>
        </div>
        <p className="text-sm text-gray-700">
          FI-AdaBoost demonstrates significant improvements across all metrics, validating the effectiveness 
          of feature-aware weighting in solar energy forecasting.
        </p>
      </div>

      {/* Methodology */}
      <div className="bg-white rounded-xl p-6 shadow-md">
        <h3 className="text-lg mb-4 flex items-center gap-2">
          <GitBranch className="w-5 h-5 text-purple-500" />
          Methodology Highlights
        </h3>
        <div className="space-y-3">
          <div className="border-l-4 border-blue-500 pl-4">
            <h4 className="text-sm mb-1">Data Sources</h4>
            <p className="text-sm text-gray-700">
              NASA POWER (365 daily records, 2024) • OpenStreetMap (10,000 building footprints)
            </p>
          </div>

          <div className="border-l-4 border-orange-500 pl-4">
            <h4 className="text-sm mb-1">Feature Engineering</h4>
            <p className="text-sm text-gray-700">
              Temporal (month sin/cos, season) • Meteorological (sunshine hours, clear sky ratio) • 
              Topographical (Solar Exposure Index: orientation, area, shading, tilt)
            </p>
          </div>

          <div className="border-l-4 border-green-500 pl-4">
            <h4 className="text-sm mb-1">Model Training</h4>
            <p className="text-sm text-gray-700">
              Bayesian optimization (Optuna) • TimeSeriesSplit validation • Decision tree weak learners (max depth: 3)
            </p>
          </div>

          <div className="border-l-4 border-purple-500 pl-4">
            <h4 className="text-sm mb-1">Evaluation</h4>
            <p className="text-sm text-gray-700">
              RMSE, MAE, R² • Diebold-Mariano statistical test • 95% confidence intervals
            </p>
          </div>
        </div>
      </div>

      {/* Impact */}
      <div className="bg-gradient-to-br from-green-50 to-emerald-50 border border-green-200 rounded-xl p-6 shadow-md">
        <h3 className="text-lg mb-4 flex items-center gap-2">
          <Users className="w-5 h-5 text-green-600" />
          Societal Impact
        </h3>
        <div className="space-y-3 text-sm text-gray-700">
          <p>
            This research supports <strong>SDG 7: Affordable and Clean Energy</strong> and <strong>SDG 13: Climate Action</strong> 
            by providing data-driven tools that can guide informed decision-making toward renewable energy adoption.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-3">
            <div className="bg-white rounded-lg p-3 border border-green-200">
              <p className="text-xs text-green-700 mb-1">Philippines Solar Potential</p>
              <p className="text-xl text-green-800">91,000 MW</p>
              <p className="text-xs text-gray-600">Feasible rooftop capacity</p>
            </div>
            <div className="bg-white rounded-lg p-3 border border-green-200">
              <p className="text-xs text-green-700 mb-1">Davao City Utilization</p>
              <p className="text-xl text-green-800">0.16%</p>
              <p className="text-xs text-gray-600">Of rooftop area (4564.9 ha)</p>
            </div>
          </div>
          <p className="mt-3">
            By optimizing solar energy forecasting, this research enables communities and institutions to better allocate 
            resources, reduce reliance on fossil fuels, and foster environmental sustainability.
          </p>
        </div>
      </div>

      {/* Footer */}
      <div className="bg-gray-50 border border-gray-200 rounded-xl p-6 text-center">
        <p className="text-sm text-gray-700">
          <strong>Thesis Project</strong> • CCS Research Agenda • Davao City, Philippines
        </p>
        <p className="text-xs text-gray-600 mt-2">
          Contributing to renewable energy forecasting through machine learning optimization
        </p>
      </div>
    </div>
  );
}
