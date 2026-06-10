import { notFound } from 'next/navigation';
import Link from 'next/link';
import { Metadata } from 'next';
import { getPageConfig, getMarkdownContent } from '@/lib/content';
import { getConfig } from '@/lib/config';
import { getRuntimeI18nConfig } from '@/lib/i18n/config';
import DynamicPageClient, { type DynamicPageLocaleData } from '@/components/pages/DynamicPageClient';
import { BasePageConfig, CardPageConfig, TextPageConfig } from '@/types/page';

// Blog posts are stored as content/blog/<post>.toml (+ .md). The list of posts is
// driven by the card items configured in content/blog.toml so there is a single
// source of truth for both the index and the generated routes.
function getPostSlugs(): string[] {
  const blogConfig = getPageConfig('blog') as BasePageConfig | null;
  if (!blogConfig || blogConfig.type !== 'card') {
    return [];
  }
  const slugs = (blogConfig as CardPageConfig).items
    .map((item) => item.href?.split('/').filter(Boolean).pop())
    .filter((slug): slug is string => Boolean(slug));
  return Array.from(new Set(slugs));
}

function loadPost(post: string, locale?: string): DynamicPageLocaleData | null {
  const pageConfig = getPageConfig(`blog/${post}`, locale) as BasePageConfig | null;
  if (!pageConfig || pageConfig.type !== 'text') {
    return null;
  }
  const textConfig = pageConfig as TextPageConfig;
  return { type: 'text', config: textConfig, content: getMarkdownContent(textConfig.source, locale) };
}

export function generateStaticParams() {
  return getPostSlugs().map((post) => ({ post }));
}

export async function generateMetadata({ params }: { params: Promise<{ post: string }> }): Promise<Metadata> {
  const { post } = await params;
  const pageConfig = getPageConfig(`blog/${post}`) as BasePageConfig | null;
  if (!pageConfig) {
    return {};
  }
  return { title: pageConfig.title, description: pageConfig.description };
}

export default async function BlogPostPage({ params }: { params: Promise<{ post: string }> }) {
  const { post } = await params;

  const baseConfig = getConfig();
  const runtimeI18n = getRuntimeI18nConfig(baseConfig.i18n);
  const targetLocales = runtimeI18n.enabled ? runtimeI18n.locales : [runtimeI18n.defaultLocale];

  const dataByLocale: Record<string, DynamicPageLocaleData> = {};
  for (const locale of targetLocales) {
    const data = loadPost(post, locale);
    if (data) {
      dataByLocale[locale] = data;
    }
  }

  const defaultData = loadPost(post);
  if (defaultData) {
    dataByLocale[runtimeI18n.defaultLocale] = dataByLocale[runtimeI18n.defaultLocale] || defaultData;
  }

  if (Object.keys(dataByLocale).length === 0) {
    notFound();
  }

  return (
    <div>
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pt-12 -mb-6">
        <Link
          href="/blog"
          className="inline-flex items-center gap-1 text-sm font-medium text-neutral-500 hover:text-accent transition-colors duration-200"
        >
          <span aria-hidden>←</span> Back to Blog
        </Link>
      </div>
      <DynamicPageClient dataByLocale={dataByLocale} defaultLocale={runtimeI18n.defaultLocale} />
    </div>
  );
}
