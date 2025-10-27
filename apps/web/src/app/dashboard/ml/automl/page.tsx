'use client'

import { Brain, Sparkles, Settings, Zap } from 'lucide-react'

export default function AutoMLPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Automated Machine Learning</h2>
        <p className="text-sm text-gray-600 mt-1">
          Automated model selection, hyperparameter tuning, and optimization
        </p>
      </div>

      {/* Coming Soon Card */}
      <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg border-2 border-blue-200 p-12">
        <div className="max-w-2xl mx-auto text-center">
          <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
            <Brain className="w-10 h-10 text-white" />
          </div>

          <h3 className="text-2xl font-bold text-gray-900 mb-4">
            AutoML Coming Soon
          </h3>

          <p className="text-gray-700 mb-8 leading-relaxed">
            Automated machine learning capabilities are currently in development.
            This feature will enable automatic model selection, hyperparameter optimization,
            and neural architecture search for your semiconductor manufacturing processes.
          </p>

          {/* Feature Preview */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
            <div className="bg-white rounded-lg p-6 border border-blue-100">
              <Sparkles className="w-8 h-8 text-blue-600 mx-auto mb-3" />
              <h4 className="font-semibold text-sm mb-2">Auto Model Selection</h4>
              <p className="text-xs text-gray-600">
                Automatically find the best algorithm for your data
              </p>
            </div>

            <div className="bg-white rounded-lg p-6 border border-blue-100">
              <Settings className="w-8 h-8 text-blue-600 mx-auto mb-3" />
              <h4 className="font-semibold text-sm mb-2">Hyperparameter Tuning</h4>
              <p className="text-xs text-gray-600">
                Optimize model parameters automatically
              </p>
            </div>

            <div className="bg-white rounded-lg p-6 border border-blue-100">
              <Zap className="w-8 h-8 text-blue-600 mx-auto mb-3" />
              <h4 className="font-semibold text-sm mb-2">Neural Architecture Search</h4>
              <p className="text-xs text-gray-600">
                Design optimal neural networks automatically
              </p>
            </div>
          </div>

          {/* CTA */}
          <div className="mt-8">
            <button
              disabled
              className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium opacity-50 cursor-not-allowed"
            >
              Coming in Next Release
            </button>
          </div>
        </div>
      </div>

      {/* Info Box */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h4 className="font-semibold text-gray-900 mb-3">What to expect:</h4>
        <ul className="space-y-2 text-sm text-gray-700">
          <li className="flex items-start gap-2">
            <span className="text-blue-600 mt-0.5">•</span>
            <span>Automated pipeline construction and optimization</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 mt-0.5">•</span>
            <span>Intelligent feature engineering and selection</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 mt-0.5">•</span>
            <span>Multi-objective optimization for accuracy and performance</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-blue-600 mt-0.5">•</span>
            <span>Experiment tracking and model versioning</span>
          </li>
        </ul>
      </div>
    </div>
  )
}
