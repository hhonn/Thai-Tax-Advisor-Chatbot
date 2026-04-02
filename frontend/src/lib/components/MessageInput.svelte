<script lang="ts">
	interface Props {
		onSend?: (message: string) => void;
	}

	let { onSend }: Props = $props();

	let inputValue = $state('');

	function handleSend() {
		const trimmed = inputValue.trim();
		if (!trimmed) return;
		onSend?.(trimmed);
		inputValue = '';
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			handleSend();
		}
	}
</script>

<div
	class="from-background via-background/95 fixed bottom-0 left-0 z-40 w-full bg-gradient-to-t to-transparent px-4 pt-4 pb-8"
>
	<div class="mx-auto max-w-3xl">
		<div
			class="bg-surface-container-lowest border-outline-variant/15 relative flex items-center rounded-full border p-2 shadow-[0_8px_32px_rgba(38,24,19,0.08)] backdrop-blur-xl"
		>
			<button class="text-on-surface-variant hover:text-primary p-4 transition-colors">
				<span class="material-symbols-outlined">add_circle</span>
			</button>
			<input
				class="text-on-surface flex-grow border-none bg-transparent py-4 text-lg placeholder:text-stone-400 focus:ring-0"
				placeholder="Share your thoughts..."
				type="text"
				bind:value={inputValue}
				onkeydown={handleKeydown}
			/>
			<button
				class="from-primary to-primary-container bg-primary text-on-primary flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br shadow-md transition-all active:scale-95"
				onclick={handleSend}
			>
				<span
					class="material-symbols-outlined"
					style="font-variation-settings: 'FILL' 0, 'wght' 600;"
				>
					arrow_upward
				</span>
			</button>
		</div>
	</div>
</div>
