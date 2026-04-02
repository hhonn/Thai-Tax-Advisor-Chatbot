<script lang="ts">
	import { browser } from '$app/environment';
	import NavBar from '$lib/components/NavBar.svelte';
	import ChatMessage from '$lib/components/ChatMessage.svelte';
	import MessageInput from '$lib/components/MessageInput.svelte';
	import ItemButton from '$lib/components/ItemButton.svelte';

	// data
	import quickAskPreset from '$data/quickAskPreset.json';
	import ChatMessageSkeleton from '$lib/components/ChatMessageSkeleton.svelte';

	type ChatEntry = {
		id: number;
		role: 'ai' | 'user';
		isWaiting?: boolean;
		message: string;
		timestamp: string;
	};

	let chatArray: ChatEntry[] = $state([]);
	let db: IDBDatabase | null = $state(null);

	$effect(() => {
		if (!browser) return;

		const request = indexedDB.open('thai-tax-advisor', 1);

		request.onupgradeneeded = (event) => {
			const database = (event.target as IDBOpenDBRequest).result;
			if (!database.objectStoreNames.contains('chats')) {
				database.createObjectStore('chats', { keyPath: 'id' });
			}
		};

		request.onsuccess = (event) => {
			db = (event.target as IDBOpenDBRequest).result;

			const transaction = db.transaction('chats', 'readonly');
			const store = transaction.objectStore('chats');
			const getAllRequest = store.getAll();

			getAllRequest.onsuccess = () => {
				chatArray = getAllRequest.result as ChatEntry[];
			};

			getAllRequest.onerror = () => {
				console.error('Failed to load chat history from IndexedDB');
			};
		};

		request.onerror = () => {
			console.error('Failed to open IndexedDB');
		};

		return () => {
			db?.close();
			db = null;
		};
	});

	// ── IndexedDB CRUD ────────────────────────────────────────────────────────

	function putChat(chat: ChatEntry) {
		chatArray = [...chatArray, chat];

		if (!db) return;
		const store = db.transaction('chats', 'readwrite').objectStore('chats');
		store.add(chat).onerror = () => {
			console.error('Failed to add chat to IndexedDB:', chat);
		};
	}

	function updateChat(id: number, updatedFields: Partial<Omit<ChatEntry, 'id'>>) {
		chatArray = chatArray.map((chat) => (chat.id === id ? { ...chat, ...updatedFields } : chat));

		if (!db) return;
		const updated = chatArray.find((chat) => chat.id === id);
		if (!updated) return;

		const store = db.transaction('chats', 'readwrite').objectStore('chats');
		store.put(updated).onerror = () => {
			console.error('Failed to update chat in IndexedDB with id:', id);
		};
	}

	function deleteChat(id: number) {
		chatArray = chatArray.filter((chat) => chat.id !== id);

		if (!db) return;
		const store = db.transaction('chats', 'readwrite').objectStore('chats');
		store.delete(id).onerror = () => {
			console.error('Failed to delete chat from IndexedDB with id:', id);
		};
	}

	// ─────────────────────────────────────────────────────────────────────────

	// random preset n item
	const randomItemN = 3;
	const randomPreset = quickAskPreset.sort(() => 0.5 - Math.random()).slice(0, randomItemN);

	function handleSend(message: string) {
		// TODO: wire up to backend API
		putChat({
			id: Date.now(),
			role: 'user',
			message,
			timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
		});
	}
</script>

<svelte:head>
	<title>ที่ปรึกษาภาษีของไทย</title>
</svelte:head>

<NavBar />

<main class="min-h-screen px-4 pt-24 pb-32 md:px-8">
	<div class="mx-auto max-w-3xl">
		<!-- Welcome Header -->
		<header class="pt-20 pb-16 text-center">
			<h1 class="mb-4 font-headline text-5xl font-medium tracking-tight text-on-surface">
				ที่ปรึกษาภาษีของไทย
			</h1>
			<p class="text-lg text-on-surface-variant">
				ยินดีต้อนรับสู่ที่ปรึกษาภาษีของไทย
				แชทกับผู้เชี่ยวชาญด้านภาษีของเราเพื่อรับคำแนะนำส่วนบุคคลและข้อมูลเชิงลึกเกี่ยวกับการวางแผนภาษีในประเทศไทย
			</p>
		</header>

		<!-- Quick Ask Preset -->
		{#if chatArray.length === 0}
			<div class="flex flex-col gap-4">
				<h2 class="font-headline text-2xl font-medium text-on-surface">คำถามด่วน</h2>
				{#each randomPreset as item, _ (item.id)}
					<ItemButton
						title={item.title}
						description={item.description}
						onAction={() => handleSend(item.prompt)}
					/>
					<!-- <Item
            title={item.title}
            description={item.description}
            /> -->
				{/each}
			</div>
		{:else}
			<!-- Chat Thread -->
			<div class="flex flex-col space-y-8">
				{#each chatArray as chat (chat.id)}
					{#if chat.isWaiting}
						<ChatMessageSkeleton role={chat.role} />
					{:else}
						<ChatMessage role={chat.role} timestamp={chat.timestamp}>
							<p class="text-lg leading-relaxed">{chat.message}</p>
						</ChatMessage>
					{/if}
				{/each}

				<!-- <ChatMessage role="user" timestamp="10:25 AM">
					<p class="text-lg leading-relaxed">
						I want to focus on the "Warm Intellectual" theme. Can we emphasize tonal layering
						instead of hard lines in the user interface?
					</p>
				</ChatMessage>

				<ChatMessage role="ai" timestamp="10:26 AM">
					<p class="text-lg leading-relaxed">
						Precisely. By utilizing shifts in hex values—like placing a
						<span class="font-semibold">surface-container-lowest</span> element on a
						<span class="font-semibold">surface-container-low</span> background—we achieve a sophisticated
						sense of elevation without the visual clutter of 1px borders.
					</p>
					<div class="mt-6 flex flex-wrap gap-2">
						<button
							class="rounded-lg bg-surface-container-low px-4 py-2 text-sm text-on-surface-variant transition-colors hover:bg-surface-container-highest"
						>
							Explore Color Specs
						</button>
						<button
							class="rounded-lg bg-surface-container-low px-4 py-2 text-sm text-on-surface-variant transition-colors hover:bg-surface-container-highest"
						>
							View Typography
						</button>
					</div>
				</ChatMessage> -->
			</div>
			<!-- <TonalCard
					imageSrc="https://lh3.googleusercontent.com/aida-public/AB6AXuBevq2b-rhuq5EibJvKS6XbhPifJy7gNp79amDR2EswppUn5BFfvGg1EOKpNhXlwWtUFdFHH9MbH8R-kPjWfmW1gjKKllpzPhEd7Jf9dYrcSAyEIHerZ8oXV5F3AuDd19I8dHnya_W7YKuwURfyV0b2FxTbVqOQvXxYOA_Qev-jr9gK4f8Ua80Phs2p0agzdWT3nVDfX3cQ3407eFlNgRJbqd8PNJVbdP51vTnUjOVRDgneuRMV0ymiGD0KLk68scdAmU40CZs9tX1p"
					imageAlt="Modern minimalist workspace with warm wood desk, high-end stationery, and soft ambient sunlight through a window"
					title="Tonal Depth Study"
					description="Visualizing how diffused light interacts with layered surfaces in a high-end editorial layout."
					actionLabel="Refine Visuals"
				/> -->
		{/if}
	</div>
</main>

<MessageInput onSend={handleSend} />
