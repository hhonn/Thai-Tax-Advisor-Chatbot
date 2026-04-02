<script lang="ts">
	import type { HTMLAnchorAttributes } from 'svelte/elements';

	type Props = {
		icon?: string;
		label: string;
		danger?: boolean;
		href: string;
	} & Omit<HTMLAnchorAttributes, 'href'>;

	let { icon, label, href, danger = false, ...restProps }: Props = $props();
</script>

<li role="menuitem">
	<!-- eslint-disable svelte/no-navigation-without-resolve -->
	<a
		{href}
		class="flex w-full items-center gap-3 px-4 py-3 text-sm transition-colors
			{danger
			? 'text-error hover:bg-error-container/30'
			: 'text-on-surface hover:bg-surface-container-high'}"
		{...restProps}
	>
		{#if icon}
			<span
				class="material-symbols-outlined text-base {danger
					? 'text-error'
					: 'text-on-surface-variant'}">{icon}</span
			>
		{/if}
		{label}
	</a>
	<!-- eslint-enable svelte/no-navigation-without-resolve -->
</li>
