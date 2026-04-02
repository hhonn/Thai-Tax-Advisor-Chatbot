<script lang="ts">
	interface Props {
		role?: 'ai' | 'user';
	}

	let { role = 'ai' }: Props = $props();
</script>

{#if role === 'ai'}
	<div class="group flex max-w-[85%] flex-col items-start gap-2">
		<div
			class="shadow-sms space-y-3 rounded-xl rounded-bl-lg bg-surface-container-high p-6 text-on-surface"
		>
			<div class="h-4 w-64 animate-pulse rounded-full bg-surface-container-highest"></div>
			<div class="h-4 w-80 animate-pulse rounded-full bg-surface-container-highest"></div>
			<div class="h-4 w-48 animate-pulse rounded-full bg-surface-container-highest"></div>
		</div>
		<div class="mx-2 h-3 w-24 animate-pulse rounded-full bg-surface-container-highest"></div>
	</div>
{:else}
	<div class="group flex max-w-[85%] flex-col items-end gap-2 self-end">
		<div
			class="space-y-3 rounded-xl rounded-br-lg bg-gradient-to-br from-primary to-primary-container p-6 text-on-primary shadow-md"
		>
			<div class="h-4 w-56 animate-pulse rounded-full bg-primary/20"></div>
			<div class="h-4 w-72 animate-pulse rounded-full bg-primary/20"></div>
		</div>
		<div class="mx-2 h-3 w-24 animate-pulse rounded-full bg-surface-container-highest"></div>
	</div>
{/if}
