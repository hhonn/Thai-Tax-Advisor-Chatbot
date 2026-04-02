<script lang="ts">
	interface Props {
		imageSrc: string;
		imageAlt: string;
		title: string;
		description: string;
		actionLabel: string;
		onAction?: () => void;
	}

	let { imageSrc, imageAlt, title, description, actionLabel, onAction }: Props = $props();
</script>

<div class="w-full py-8">
	<div
		class="bg-surface-container-lowest flex flex-col items-center overflow-hidden rounded-xl shadow-[0_32px_64px_-12px_rgba(38,24,19,0.06)] md:flex-row"
	>
		<img class="h-64 w-full object-cover md:w-1/2" src={imageSrc} alt={imageAlt} />
		<div class="p-8 md:w-1/2 md:p-12">
			<h3 class="font-headline mb-3 text-2xl font-medium">{title}</h3>
			<p class="text-on-surface-variant mb-6">{description}</p>
			<button
				class="text-primary group flex cursor-pointer items-center gap-2 font-medium"
				onclick={onAction}
			>
				<span>{actionLabel}</span>
				<span class="material-symbols-outlined transition-transform group-hover:translate-x-1">
					arrow_forward
				</span>
			</button>
		</div>
	</div>
</div>
