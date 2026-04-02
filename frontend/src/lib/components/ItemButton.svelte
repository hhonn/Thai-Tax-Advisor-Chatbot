<script lang="ts">
	import { cn } from '$lib/utils/cn';

	interface Props {
		icon?: string;
		title: string;
		description?: string;
		onAction?: () => void;
	}

	let { icon, title, description, onAction }: Props = $props();
</script>

<button
	class={cn(
		'flex items-center gap-4 overflow-hidden rounded-xl bg-surface-container-lowest p-5 shadow-[0_32px_64px_-12px_rgba(38,24,19,0.06)] md:flex-row text-left',
		onAction ? 'cursor-pointer hover:bg-surface-container-low' : ''
	)}
	onclick={onAction}
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
	
</button>
