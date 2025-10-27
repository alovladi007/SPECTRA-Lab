'use client'

import { Eye, BarChart3, GitBranch, FileText } from 'lucide-react'

export default function ExplainabilityPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Model Explainability & Interpretability</h2>
        <p className="text-sm text-gray-600 mt-1">
          Understand and interpret ML model predictions with SHAP, LIME, and feature analysis
        </p>
      </div>

      {/* Coming Soon Card */}
      <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg border-2 border-purple-200 p-12">
        <div className="max-w-2xl mx-auto text-center">
          <div className="w-20 h-20 bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
            <Eye className="w-10 h-10 text-white" />
          </div>

          <h3 className="text-2xl font-bold text-gray-900 mb-4">
            Model Explainability Coming Soon
          </h3>

          <p className="text-gray-700 mb-8 leading-relaxed">
            Advanced model interpretability features are currently in development.
            This module will provide deep insights into how your ML models make predictions,
            helping you build trust and meet regulatory requirements.
          </p>

          {/* Feature Preview */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-8">
            <div className="bg-white rounded-lg p-6 border border-purple-100">
              <BarChart3 className="w-8 h-8 text-purple-600 mx-auto mb-3" />
              <h4 className="font-semibold text-sm mb-2">SHAP Values</h4>
              <p className="text-xs text-gray-600">
                Understand feature contributions to individual predictions
              </p>
            </div>

            <div className="bg-white rounded-lg p-6 border border-purple-100">
              <GitBranch className="w-8 h-8 text-purple-600 mx-auto mb-3" />
              <h4 className="font-semibold text-sm mb-2">LIME Analysis</h4>
              <p className="text-xs text-gray-600">
                Local interpretable model-agnostic explanations
              </p>
            </div>

            <div className="bg-white rounded-lg p-6 border border-purple-100">
              <FileText className="w-8 h-8 text-purple-600 mx-auto mb-3" />
              <h4 className="font-semibold text-sm mb-2">Feature Interactions</h4>
              <p className="text-xs text-gray-600">
                Discover how features work together
              </p>
            </div>

            <div className="bg-white rounded-lg p-6 border border-purple-100">
              <Eye className="w-8 h-8 text-purple-600 mx-auto mb-3" />
              <h4 className="font-semibold text-sm mb-2">Decision Paths</h4>
              <p className="text-xs text-gray-600">
                Visualize the decision-making process
              </p>
            </div>
          </div>

          {/* CTA */}
          <div className="mt-8">
            <button
              disabled
              className="px-6 py-3 bg-purple-600 text-white rounded-lg font-medium opacity-50 cursor-not-allowed"
            >
              Coming in Next Release
            </button>
          </div>
        </div>
      </div>

      {/* Info Box */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h4 className="font-semibold text-gray-900 mb-3">Planned Features:</h4>
        <ul className="space-y-2 text-sm text-gray-700">
          <li className="flex items-start gap-2">
            <span className="text-purple-600 mt-0.5">•</span>
            <span>SHAP (SHapley Additive exPlanations) value visualization</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-purple-600 mt-0.5">•</span>
            <span>LIME (Local Interpretable Model-agnostic Explanations) analysis</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-purple-600 mt-0.5">•</span>
            <span>Partial dependence plots and ICE curves</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-purple-600 mt-0.5">•</span>
            <span>Feature interaction detection and visualization</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-purple-600 mt-0.5">•</span>
            <span>Decision tree visualization for ensemble models</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-purple-600 mt-0.5">•</span>
            <span>Automated report generation for regulatory compliance</span>
          </li>
        </ul>
      </div>
    </div>
  )
}
