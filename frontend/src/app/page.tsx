'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  ChatBubbleLeftRightIcon, 
  CpuChipIcon, 
  EyeIcon, 
  SpeakerWaveIcon,
  SparklesIcon,
  RocketLaunchIcon,
  ChartBarIcon,
  ShieldCheckIcon
} from '@heroicons/react/24/outline';
import { ChatInterface } from '@/components/ChatInterface';
import { ModelDashboard } from '@/components/ModelDashboard';
import { MultiModalInterface } from '@/components/MultiModalInterface';
import { SystemMonitor } from '@/components/SystemMonitor';

const features = [
  {
    name: 'RetNet Architecture',
    description: 'Revolutionary retention mechanism for efficient long-sequence modeling',
    icon: CpuChipIcon,
    color: 'text-blue-600',
    bgColor: 'bg-blue-100',
  },
  {
    name: 'Mixture of Experts',
    description: 'Dynamic routing for scalable model architectures with load balancing',
    icon: ChartBarIcon,
    color: 'text-green-600',
    bgColor: 'bg-green-100',
  },
  {
    name: 'Multi-Modal Fusion',
    description: 'Advanced fusion of text, vision, and audio with cross-modal attention',
    icon: EyeIcon,
    color: 'text-purple-600',
    bgColor: 'bg-purple-100',
  },
  {
    name: 'Real-Time Streaming',
    description: 'WebSocket-based streaming inference with low latency',
    icon: SpeakerWaveIcon,
    color: 'text-orange-600',
    bgColor: 'bg-orange-100',
  },
  {
    name: 'Advanced Training',
    description: 'LoRA, QLoRA, and gradient checkpointing for efficient training',
    icon: RocketLaunchIcon,
    color: 'text-red-600',
    bgColor: 'bg-red-100',
  },
  {
    name: 'Enterprise Ready',
    description: 'Production-grade security, monitoring, and scalability',
    icon: ShieldCheckIcon,
    color: 'text-indigo-600',
    bgColor: 'bg-indigo-100',
  },
];

const tabs = [
  { name: 'Chat', id: 'chat', icon: ChatBubbleLeftRightIcon },
  { name: 'Models', id: 'models', icon: CpuChipIcon },
  { name: 'Multi-Modal', id: 'multimodal', icon: EyeIcon },
  { name: 'Monitor', id: 'monitor', icon: ChartBarIcon },
];

export default function HomePage() {
  const [activeTab, setActiveTab] = useState('chat');

  const renderTabContent = () => {
    switch (activeTab) {
      case 'chat':
        return <ChatInterface />;
      case 'models':
        return <ModelDashboard />;
      case 'multimodal':
        return <MultiModalInterface />;
      case 'monitor':
        return <SystemMonitor />;
      default:
        return <ChatInterface />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg">
                <SparklesIcon className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">NeuroForge</h1>
                <p className="text-sm text-gray-500">Advanced AI Platform</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="hidden sm:flex items-center space-x-2 text-sm text-gray-500">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>System Online</span>
              </div>
              <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                Get Started
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-16 bg-gradient-to-r from-blue-600 to-purple-600 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-4xl md:text-6xl font-bold mb-6">
              Next-Generation
              <span className="block text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 to-orange-400">
                AI Platform
              </span>
            </h1>
            <p className="text-xl md:text-2xl mb-8 text-blue-100 max-w-3xl mx-auto">
              Experience the future of AI with cutting-edge transformer architectures,
              multi-modal processing, and real-time streaming capabilities.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="px-8 py-3 bg-white text-blue-600 rounded-lg font-semibold hover:bg-gray-100 transition-colors">
                Start Building
              </button>
              <button className="px-8 py-3 border-2 border-white text-white rounded-lg font-semibold hover:bg-white hover:text-blue-600 transition-colors">
                View Demo
              </button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Revolutionary AI Technologies
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              NeuroForge combines the latest research in AI with production-ready
              infrastructure for building next-generation applications.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white p-6 rounded-xl shadow-lg hover:shadow-xl transition-shadow border border-gray-100"
              >
                <div className={`w-12 h-12 ${feature.bgColor} rounded-lg flex items-center justify-center mb-4`}>
                  <feature.icon className={`w-6 h-6 ${feature.color}`} />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  {feature.name}
                </h3>
                <p className="text-gray-600">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Main Interface */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="bg-white rounded-xl shadow-lg overflow-hidden">
            {/* Tab Navigation */}
            <div className="border-b border-gray-200">
              <nav className="flex space-x-8 px-6">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                      activeTab === tab.id
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    <tab.icon className="w-5 h-5" />
                    <span>{tab.name}</span>
                  </button>
                ))}
              </nav>
            </div>

            {/* Tab Content */}
            <div className="p-6">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3 }}
              >
                {renderTabContent()}
              </motion.div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="col-span-1 md:col-span-2">
              <div className="flex items-center space-x-3 mb-4">
                <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg">
                  <SparklesIcon className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-bold">NeuroForge</h3>
                  <p className="text-gray-400">Advanced AI Platform</p>
                </div>
              </div>
              <p className="text-gray-400 mb-4">
                Building the future of AI with cutting-edge technologies and
                production-ready infrastructure.
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Platform</h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">Models</a></li>
                <li><a href="#" className="hover:text-white transition-colors">API</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Documentation</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Examples</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Support</h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">Help Center</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Community</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Contact</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Status</a></li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2024 NeuroForge. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
