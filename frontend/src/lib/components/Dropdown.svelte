<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		trigger: Snippet;
		children: Snippet;
		align?: 'left' | 'right';
	}

	let { trigger, children, align = 'left' }: Props = $props();

	let open = $state(false);
	let rootEl: HTMLDivElement;

	function toggle() {
		open = !open;
	}

	function handleWindowClick(event: MouseEvent) {
		if (open && rootEl && !rootEl.contains(event.target as Node)) {
			open = false;
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Escape') open = false;
	}
</script>

<svelte:window onclick={handleWindowClick} onkeydown={handleKeydown} />

<div bind:this={rootEl} class="relative inline-block">
	<!-- Trigger -->
	<div
		role="button"
		tabindex="0"
		aria-haspopup="true"
		aria-expanded={open}
		onclick={toggle}
		onkeydown={(e) => e.key === 'Enter' || e.key === ' ' ? toggle() : null}
	>
		{@render trigger()}
	</div>

	<!-- Dropdown Card -->
	{#if open}
		<div
			class="absolute top-full z-50 mt-2 min-w-56 overflow-hidden rounded-xl border border-outline-variant/20 bg-surface-container-lowest shadow-[0_8px_32px_rgba(38,24,19,0.10)]
			{align === 'right' ? 'right-0' : 'left-0'}"
			role="menu"
		>
			<ul class="flex flex-col py-2">
				{@render children()}
			</ul>
		</div>
	{/if}
</div>
