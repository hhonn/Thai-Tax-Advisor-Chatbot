<script lang="ts">
	import type { HTMLButtonAttributes } from 'svelte/elements';

	type Props = {
		icon?: string;
		label: string;
		danger?: boolean;
		onclick?: () => void;
	} & Omit<HTMLButtonAttributes, 'onclick' | 'type'>;

	let { icon, label, onclick, danger = false, ...restProps }: Props = $props();
</script>

<li role="menuitem">
	<button
		type="button"
		{onclick}
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
	</button>
</li>
