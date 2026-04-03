<script lang="ts">
	import { browser } from '$app/environment';
	import { marked } from 'marked';
	import DOMPurify from 'dompurify';

	interface Props {
		source: string;
	}

	let { source }: Props = $props();

	// GitHub-flavoured markdown + line-break-to-<br>
	marked.use({ gfm: true, breaks: true });

	let html = $derived(
		browser
			? DOMPurify.sanitize(String(marked.parse(source)), { USE_PROFILES: { html: true } })
			: String(marked.parse(source))
	);
</script>

<!--
  prose       — Tailwind Typography base styles
  prose-sm    — compact size for chat bubbles
  max-w-none  — fill the parent bubble width
-->
<div
	class="prose prose-sm max-w-none"
	style="
    --tw-prose-body:          var(--color-on-surface);
    --tw-prose-headings:      var(--color-on-surface);
    --tw-prose-lead:          var(--color-on-surface-variant);
    --tw-prose-links:         var(--color-primary);
    --tw-prose-bold:          var(--color-on-surface);
    --tw-prose-counters:      var(--color-on-surface-variant);
    --tw-prose-bullets:       var(--color-on-surface-variant);
    --tw-prose-hr:            var(--color-outline-variant);
    --tw-prose-quotes:        var(--color-on-surface-variant);
    --tw-prose-quote-borders: var(--color-primary);
    --tw-prose-captions:      var(--color-on-surface-variant);
    --tw-prose-code:          var(--color-on-surface);
    --tw-prose-pre-code:      var(--color-on-surface);
    --tw-prose-pre-bg:        var(--color-surface-container-highest);
    --tw-prose-th-borders:    var(--color-outline-variant);
    --tw-prose-td-borders:    var(--color-outline-variant);
  "
>
	<!-- eslint-disable-next-line svelte/no-at-html-tags -->
	{@html html}
</div>

