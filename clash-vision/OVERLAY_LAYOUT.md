# Overlay Layout Changes

## Fixed Issues
1. **Removed next card prediction from overlay** - No longer displayed as visual card next to current hand
2. **Repositioned detection stats** - Moved from top-left to top-right (below elixir bar) to avoid covering game cards
3. **Improved AI predictions layout** - Now displays below current hand in horizontal format with confidence percentage

## Current Overlay Layout

### Top Left Corner
- Performance stats (FPS, API interval, next call timer)
- NMS filtering statistics when active

### Top Right Corner  
- Opponent elixir bar
- Detection statistics (visible cards count, NMS status)
- Opponent deck (8 cards in 2x4 grid)

### Bottom Left Corner
- Current hand (4 cards with images)
- AI predictions (horizontal list below hand)

### Bottom Right Corner
- Game information (last played card, total plays, match time)

## Benefits
- No overlay elements cover actual game cards
- Clean separation of UI elements
- AI predictions still visible but not intrusive
- Better use of screen real estate
- Maintained visual hierarchy and readability