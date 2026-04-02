<script lang="ts">
	interface Props {
		icon?: string;
		title: string;
		description?: string;
		actionLabel?: string;
		onAction?: () => void;
	}

	let { icon, title, description, actionLabel, onAction }: Props = $props();
</script>

<div
	class="flex items-center gap-4 overflow-hidden rounded-xl bg-surface-container-lowest p-5 shadow-[0_32px_64px_-12px_rgba(38,24,19,0.06)] md:flex-row"
>
	{#if icon}
		<div
			class="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-surface-container-high text-primary"
		>
			<span class="material-symbols-outlined">{icon}</span>
		</div>
	{/if}

	<div class="min-w-0 flex-1">
		<h4 class="font-headline text-base font-semibold text-on-surface">{title}</h4>
		{#if description}
			<p class="mt-0.5 truncate text-sm text-on-surface-variant">{description}</p>
		{/if}
	</div>

	{#if actionLabel}
		<button
			class="group flex shrink-0 cursor-pointer items-center gap-1 text-sm font-medium text-primary"
			onclick={onAction}
		>
			<span>{actionLabel}</span>
			<span class="material-symbols-outlined text-base transition-transform"> arrow_forward </span>
		</button>
	{/if}
</div>
