<script lang="ts">
	import Dropdown from '$lib/components/Dropdown.svelte';
	import DropdownItemLink from '$lib/components/DropdownItemLink.svelte';
</script>

<nav class="tonal-shift-no-lines fixed top-0 z-50 w-full bg-surface">
	<div class="mx-auto flex h-20 max-w-7xl items-center justify-between px-8">
		<div class="flex items-center gap-8">
			<span class="font-headline text-2xl font-semibold tracking-tight text-primary">
				Thai Tax Advisor
			</span>
		</div>
		<div class="flex items-center gap-4">
			<Dropdown align="right">
				{#snippet trigger()}
					<button class="bg-primary p-2 px-4 text-on-primary">
						ข้อมูลเพิ่มเติม
						<span class="material-symbols-outlined">arrow_outward</span>
					</button>
				{/snippet}

				<DropdownItemLink label="กรมสรรพากร" href="https://www.rd.go.th/" rel="external" icon="link_2" />
				<DropdownItemLink label="กรมสรรพสามิต" href="https://www.excise.go.th/" rel="external" icon="link_2" />
				<DropdownItemLink label="กรมศุลกากร" href="https://www.customs.go.th/" rel="external" icon="link_2" />
			</Dropdown>
		</div>
	</div>
</nav>
