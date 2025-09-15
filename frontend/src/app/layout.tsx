import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Providers } from './providers';
import { Toaster } from 'react-hot-toast';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'NeuroForge - Advanced AI Platform',
  description: 'Next-generation multi-modal AI platform with cutting-edge transformer architectures',
  keywords: ['AI', 'Machine Learning', 'Transformer', 'Multi-modal', 'RetNet', 'MoE'],
  authors: [{ name: 'NeuroForge Team' }],
  viewport: 'width=device-width, initial-scale=1',
  themeColor: '#3b82f6',
  openGraph: {
    title: 'NeuroForge - Advanced AI Platform',
    description: 'Next-generation multi-modal AI platform with cutting-edge transformer architectures',
    type: 'website',
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'NeuroForge - Advanced AI Platform',
    description: 'Next-generation multi-modal AI platform with cutting-edge transformer architectures',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.className} h-full bg-gray-50`}>
        <Providers>
          {children}
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
              },
            }}
          />
        </Providers>
      </body>
    </html>
  );
}
