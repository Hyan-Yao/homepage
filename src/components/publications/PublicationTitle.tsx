import Link from 'next/link';
import { Publication } from '@/types/publication';
import { getTitleLink } from '@/lib/publicationLink';

export default function PublicationTitle({ pub }: { pub: Publication }) {
    const link = getTitleLink(pub);
    const className = 'hover:text-accent transition-colors duration-200';

    if (!link) return <>{pub.title}</>;
    if (link.internal) {
        return <Link href={link.href} prefetch={true} className={className}>{pub.title}</Link>;
    }
    return <a href={link.href} target="_blank" rel="noopener noreferrer" className={className}>{pub.title}</a>;
}
