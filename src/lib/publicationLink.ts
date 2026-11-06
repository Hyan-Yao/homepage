import { Publication } from '@/types/publication';

// Where a publication's title points: project page first, then the paper itself, then its blog post.
export function getTitleLink(pub: Publication): { href: string; internal: boolean } | null {
  if (pub.projectPage) return { href: pub.projectPage, internal: false };
  if (pub.url) return { href: pub.url, internal: false };
  if (pub.arxiv) return { href: pub.arxiv, internal: false };
  if (pub.doi) return { href: `https://doi.org/${pub.doi}`, internal: false };
  if (pub.blog) return { href: pub.blog, internal: true };
  return null;
}
