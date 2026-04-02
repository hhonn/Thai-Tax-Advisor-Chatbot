<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		role: 'ai' | 'user';
		timestamp: string;
		children: Snippet;
	}

	let { role, timestamp, children }: Props = $props();
</script>

{#if role === 'ai'}
	<div class="group flex max-w-[85%] flex-col items-start gap-2">
		<div class="bg-surface-container-high text-on-surface rounded-xl rounded-bl-lg p-6 shadow-sm">
			{@render children()}
		</div>
		<span class="text-on-surface-variant px-2 text-sm">Assistant • {timestamp}</span>
	</div>
{:else}
	<div class="group flex max-w-[85%] flex-col items-end gap-2 self-end">
		<div
			class="from-primary to-primary-container text-on-primary rounded-xl rounded-br-lg bg-gradient-to-br p-6 shadow-md"
		>
			{@render children()}
		</div>
		<span class="text-on-surface-variant px-2 text-sm">You • {timestamp}</span>
	</div>
{/if}
