# Chinese translations, staged but not live

These are complete Chinese translations of the non-blog pages. They are parked
here so that switching the site to 中文 only affects the blog — every other page
falls back to the English content in `content/`.

The loader in `src/lib/content.ts` looks for a directory named `content_<locale>`,
so `content_zh_draft` is never read. Nothing else references it.

To publish them, move the files back:

    mv content_zh_draft/*.toml content_zh_draft/*.md content_zh/
    rm content_zh/README.md   # if this file gets moved along with them

Note that `blog.md` is a leftover "敬请期待" placeholder from before the blog had
Chinese content; `content_zh/blog.toml` is now a card index and no longer points
at it, so it can be deleted rather than restored.
