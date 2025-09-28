import { useState, useEffect } from "react";
import ElixirImg from "./assets/elixir.png";
import KnightImg from "./assets/knight.png";
import ValkyrieImg from "./assets/valkyrie.png";
import HogRiderImg from "./assets/hogrider.png";
import MiniPekkaImg from "./assets/minipekka.png";
import DartGoblinImg from "./assets/dartgoblin.png";
import BomberImg from "./assets/bomber.png";
import BabyDragonImg from "./assets/babydragon.png";
import "./App.css";

const cardImages = {
	Knight: KnightImg,
	Valkyrie: ValkyrieImg,
	HogRider: HogRiderImg,
	MiniPekka: MiniPekkaImg,
	DartGoblin: DartGoblinImg,
	Bomber: BomberImg,
	BabyDragon: BabyDragonImg,
};

function ElixirBar({ elixir }) {
	return (
		<div className="elixir-bar-section">
			<h2 className="elixir-header">Opponent Elixir</h2>
			<div className="elixir-bar-row">
				<img src={ElixirImg} alt="elixir" className="elixir-icon" />
				<div className="elixir-bar">
					{[...Array(10)].map((_, i) => (
						<div key={i} className={`elixir-segment${elixir > i ? " filled" : ""}`}></div>
					))}
					<span className="elixir-label">{elixir} / 10</span>
				</div>
			</div>
		</div>
	);
}

function CardList({ title, cards }) {
	return (
		<div className="card-list">
			<h3>{title}</h3>
			<div className="cards">
				{cards && cards.length > 0 ? (
					cards.map((card, i) => (
						<img src={cardImages[card]} alt={card} className="troop-img" key={i} />
					))
				) : (
					<span className="card-item">None</span>
				)}
			</div>
		</div>
	);
}

function Prediction({ prediction }) {
	return (
		<div className="prediction">
			<h3>Next Prediction</h3>
			{/* <span className="prediction-name">
				{prediction.replace(/([A-Z])/g, " $1").trim() || "N/A"}
			</span> */}
			{cardImages[prediction] && (
				<img src={cardImages[prediction]} alt={prediction} className="troop-img" />
			)}
		</div>
	);
}

// Sample data simulating backend response
/*
const sampleGameState = {
	visible_cards: ["Knight", "Valkyrie", "HogRider"],
	elixir_opponent: 6,
	next_prediction: "MiniPekka",
	deck: [
		"Knight",
		"Valkyrie",
		"HogRider",
		"MiniPekka",
		"DartGoblin",
		"Bomber",
		"BabyDragon",
	],
	current_hand: ["Knight", "Valkyrie", "HogRider", "MiniPekka"],
};
*/

// Fetch game state from backend API (using gamestate/live_inference output)
// Assumes backend endpoint returns JSON in the format:
// {
//   "visible_cards": [...],
//   "elixir_opponent": ...,
//   "next_prediction": ...,
//   "deck": [...],
//   "current_hand": [...]
// }
const API_URL = "http://localhost:5000/api/game_state"; // Change to your backend endpoint

function App() {
	const [gameState, setGameState] = useState(null);

	useEffect(() => {
		async function fetchState() {
			try {
				const res = await fetch(API_URL);
				const data = await res.json();
				setGameState(data);
			} catch (err) {
				// handle error, optionally set sampleGameState
			}
		}
		fetchState();
		const interval = setInterval(fetchState, 1000); // Poll every second
		return () => clearInterval(interval);
	}, []);

	// Use gameState if available, else fallback to sampleGameState
	const state = gameState || sampleGameState;

	return (
		<div className="App">
			<div className="top-row">
				<div className="top-left">
					<header className="main-title">Clash Quant</header>
				</div>
				<div className="top-right">
					<ElixirBar elixir={state.elixir_opponent} />
				</div>
			</div>
			<div className="main-content-row">
				<div className="left-column">
					<CardList title="Visible Cards" cards={state.visible_cards} />
					<CardList title="Current Hand" cards={state.current_hand} />
					<Prediction prediction={state.next_prediction} />
				</div>
				<div className="right-column">
					<CardList title="Opponent Deck" cards={state.deck} />
				</div>
			</div>
		</div>
	);
}

export default App;
