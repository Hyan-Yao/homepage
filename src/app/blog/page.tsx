import { notFound } from 'next/navigation';
import { Metadata } from 'next';
import { getPageConfig, getMarkdownContent } from '@/lib/content';
import { getConfig } from '@/lib/config';
import { getRuntimeI18nConfig } from '@/lib/i18n/config';
import DynamicPageClient, { type DynamicPageLocaleData } from '@/components/pages/DynamicPageClient';
import { BasePageConfig, CardPageConfig, TextPageConfig } from '@/types/page';

function loadBlogIndex(locale?: string): DynamicPageLocaleData | null {
  const pageConfig = getPageConfig('blog', locale) as BasePageConfig | null;
  if (!pageConfig) {
    return null;
  }

  if (pageConfig.type === 'card') {
    return { type: 'card', config: pageConfig as CardPageConfig };
  }

  // Fallback: allow a plain text blog landing page if configured that way.
  if (pageConfig.type === 'text') {
    const textConfig = pageConfig as TextPageConfig;
    return { type: 'text', config: textConfig, content: getMarkdownContent(textConfig.source, locale) };
  }

  return null;
}

export async function generateMetadata(): Promise<Metadata> {
  const pageConfig = getPageConfig('blog') as BasePageConfig | null;
  if (!pageConfig) {
    return {};
  }
  return { title: pageConfig.title, description: pageConfig.description };
}

export default function BlogIndexPage() {
  const baseConfig = getConfig();
  const runtimeI18n = getRuntimeI18nConfig(baseConfig.i18n);
  const targetLocales = runtimeI18n.enabled ? runtimeI18n.locales : [runtimeI18n.defaultLocale];

  const dataByLocale: Record<string, DynamicPageLocaleData> = {};
  for (const locale of targetLocales) {
    const data = loadBlogIndex(locale);
    if (data) {
      dataByLocale[locale] = data;
    }
  }

  const defaultData = loadBlogIndex();
  if (defaultData) {
    dataByLocale[runtimeI18n.defaultLocale] = dataByLocale[runtimeI18n.defaultLocale] || defaultData;
  }

  if (Object.keys(dataByLocale).length === 0) {
    notFound();
  }

  return <DynamicPageClient dataByLocale={dataByLocale} defaultLocale={runtimeI18n.defaultLocale} />;
}
