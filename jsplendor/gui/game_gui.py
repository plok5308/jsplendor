import pygame
import os
import numpy as np
from jsplendor.utils import TestLogger, Element, ActionLogger
from stable_baselines3 import PPO
from jsplendor.env.observation import get_observation
import torch
from jsplendor.utils.config import get_coin_comb
from jsplendor.utils.logger import ActionLogger
from copy import deepcopy
from jsplendor.game import Game
from jsplendor.models.random_player import RandomPlayer

class SplendorGUI:
    def __init__(self, game):
        pygame.init()
        self.game = game
        
        # Window settings
        self.WINDOW_WIDTH = 1600
        self.WINDOW_HEIGHT = 1200
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("JSplendor")

        # Colors
        self.WHITE = (240, 240, 240)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        self.COLORS = {
            'WHITE': (240, 240, 240),
            'BLUE': (0, 0, 255),
            'GREEN': (0, 255, 0),
            'RED': (255, 0, 0),
            'BLACK': (0, 0, 0),
            'GOLD': (255, 215, 0)
        }

        # Add border colors for gems
        self.GEM_BORDERS = {
            'WHITE': (128, 128, 128),
            'BLUE': (0, 0, 180),
            'GREEN': (0, 180, 0),
            'RED': (180, 0, 0),
            'BLACK': (64, 64, 64),
            'GOLD': (180, 150, 0)
        }

        # Card dimensions
        self.CARD_WIDTH = 150  # Back to original
        self.CARD_HEIGHT = 200  # Back to original
        self.CARD_MARGIN = 20

        # Add card background colors for each level
        self.CARD_COLORS = {
            1: (255, 200, 200),  # Stronger pink for level 1
            2: (200, 255, 200),  # Stronger mint green for level 2
            3: (200, 200, 255),  # Stronger light blue for level 3
        }

        # Add scrollbar settings
        self.scroll_y = 0  # Current scroll position
        self.scroll_dragging = False
        self.scroll_bar_width = 15
        self.visible_lines = 10  # Number of visible lines in log
        self.line_height = 25  # Height of each line in pixels

        # Add player labels
        self.PLAYER1_COLOR = (100, 150, 255)  # Light blue
        self.PLAYER2_COLOR = (255, 150, 100)  # Light orange

        # Load assets
        self.load_assets()

    def load_assets(self):
        # Create asset directory if it doesn't exist
        if not os.path.exists('jsplendor/gui/assets'):
            os.makedirs('jsplendor/gui/assets')

        # Initialize font
        self.font = pygame.font.Font(None, 24)

    def draw_gem(self, color, x, y, radius, surface=None):
        target_surface = surface if surface is not None else self.screen
        # Draw gem border
        pygame.draw.circle(target_surface, self.GEM_BORDERS[color], (x, y), radius)
        # Draw gem interior
        pygame.draw.circle(target_surface, self.COLORS[color], (x, y), radius - 2)

    def draw_card(self, card, x, y, surface=None):
        target_surface = surface if surface is not None else self.screen
        if card is None:
            # Draw card back instead of gray rectangle
            pygame.draw.rect(target_surface, (139, 69, 19), (x, y, self.CARD_WIDTH, self.CARD_HEIGHT))  # Brown color
            pygame.draw.rect(target_surface, (101, 67, 33), (x, y, self.CARD_WIDTH, self.CARD_HEIGHT), 2)  # Darker brown border
            # Add some decoration to card back
            pygame.draw.rect(target_surface, (101, 67, 33), 
                           (x + 10, y + 10, self.CARD_WIDTH - 20, self.CARD_HEIGHT - 20), 2)
            return

        # Draw card background with level-specific color
        bg_color = self.CARD_COLORS[card.level]
        pygame.draw.rect(target_surface, bg_color, (x, y, self.CARD_WIDTH, self.CARD_HEIGHT))
        pygame.draw.rect(target_surface, self.BLACK, (x, y, self.CARD_WIDTH, self.CARD_HEIGHT), 2)

        # Draw card name
        name_text = self.font.render(card.name, True, self.BLACK)
        target_surface.blit(name_text, (x + 5, y + 5))

        # Draw victory points
        vp_text = self.font.render(f"VP: {card.victory_point}", True, self.BLACK)
        target_surface.blit(vp_text, (x + 5, y + 25))

        # Draw gem color with border
        self.draw_gem(card.gem_color, 
                     x + self.CARD_WIDTH - 20, 
                     y + 20, 
                     12, target_surface)

        # Draw price
        y_offset = 50
        for color, amount in zip(['WHITE', 'BLUE', 'GREEN', 'RED', 'BLACK'], card.price):
            if amount > 0:
                # Draw small gem icon
                self.draw_gem(color, x + 15, y + y_offset + 8, 8, target_surface)
                # Draw amount with white text for dark backgrounds
                text_color = self.BLACK
                price_text = self.font.render(str(amount), True, text_color)
                target_surface.blit(price_text, (x + 30, y + y_offset))
                y_offset += 20

    def draw_coins(self, coins, x, y, surface=None):
        target_surface = surface if surface is not None else self.screen
        for i, (color, count) in enumerate(coins.items()):
            # Draw coin with border
            self.draw_gem(color, x + i * 60, y, 25, target_surface)
            # Draw count with white text for dark backgrounds
            text_color = self.WHITE if color in ['BLACK', 'BLUE'] else self.BLACK
            count_text = self.font.render(str(count), True, text_color)
            text_rect = count_text.get_rect(center=(x + i * 60, y))
            target_surface.blit(count_text, text_rect)

    def draw_noble(self, noble, x, y, surface=None):
        target_surface = surface if surface is not None else self.screen
        if noble:
            # Draw noble card background
            pygame.draw.rect(target_surface, (255, 223, 196), (x, y, 120, 120))
            pygame.draw.rect(target_surface, self.BLACK, (x, y, 120, 120), 2)
            
            # Draw victory points
            noble_text = self.font.render(f"Noble: {noble.victory_point}VP", True, self.BLACK)
            target_surface.blit(noble_text, (x + 10, y + 10))
            
            # Draw required gems
            y_offset = 40
            for color, amount in zip(['WHITE', 'BLUE', 'GREEN', 'RED', 'BLACK'], noble.price):
                if amount > 0:
                    # Draw small gem icon
                    self.draw_gem(color, x + 25, y + y_offset, 8, target_surface)
                    # Draw amount
                    price_text = self.font.render(str(amount), True, self.BLACK)
                    target_surface.blit(price_text, (x + 45, y + y_offset - 5))
                    y_offset += 20

    def draw_square_gem(self, color, x, y, size, surface=None):
        target_surface = surface if surface is not None else self.screen
        # Draw gem border
        border_rect = pygame.Rect(x - size//2, y - size//2, size, size)
        pygame.draw.rect(target_surface, self.GEM_BORDERS[color], border_rect)
        # Draw gem interior
        inner_rect = pygame.Rect(x - size//2 + 2, y - size//2 + 2, size - 4, size - 4)
        pygame.draw.rect(target_surface, self.COLORS[color], inner_rect)

    def draw_game_state(self, surface=None):
        """Draw the base game state (cards, coins, etc)"""
        target_surface = surface if surface is not None else self.screen
        
        # Fill background
        target_surface.fill((200, 200, 200))

        # Draw player information at top
        # Player 1 info (left side)
        p1_step_text = self.font.render(f"Player 1 - Step: {self.game.players[0].step}", True, self.BLACK)
        p1_vp_text = self.font.render(f"Victory Points: {self.game.players[0].sum_victory_point}", True, self.BLACK)
        p1_noble_text = self.font.render(f"Nobles: {len(self.game.players[0].noble_cards)}", True, self.BLACK)
        
        # Player 2 info (right side)
        p2_step_text = self.font.render(f"Player 2 - Step: {self.game.players[1].step}", True, self.BLACK)
        p2_vp_text = self.font.render(f"Victory Points: {self.game.players[1].sum_victory_point}", True, self.BLACK)
        p2_noble_text = self.font.render(f"Nobles: {len(self.game.players[1].noble_cards)}", True, self.BLACK)
        
        # Position player info
        target_surface.blit(p1_step_text, (20, 10))
        target_surface.blit(p1_vp_text, (20, 30))
        target_surface.blit(p1_noble_text, (20, 50))
        
        target_surface.blit(p2_step_text, (self.WINDOW_WIDTH - 250, 10))
        target_surface.blit(p2_vp_text, (self.WINDOW_WIDTH - 250, 30))
        target_surface.blit(p2_noble_text, (self.WINDOW_WIDTH - 250, 50))

        # Fixed y-positions for different sections
        NOBLE_Y = 80
        LEVEL3_Y = 250
        LEVEL2_Y = 500
        LEVEL1_Y = 750
        PLAYER_CARDS_Y = 1000
        
        # Board coins position on right side
        BOARD_COINS_Y = 300
        PLAYER_COINS_Y = 500

        # Draw noble cards
        for i, noble in enumerate(self.game.board.noble_cards):
            self.draw_noble(noble, 100 + i * 160, NOBLE_Y, target_surface)

        # Draw development cards
        level_positions = [
            (self.game.board.table_level3, "Level 3", LEVEL3_Y),
            (self.game.board.table_level2, "Level 2", LEVEL2_Y),
            (self.game.board.table_level1, "Level 1", LEVEL1_Y),
        ]
        
        for level_cards, label, y_pos in level_positions:
            # Draw level label
            label_text = self.font.render(label, True, self.BLACK)
            target_surface.blit(label_text, (100, y_pos - 25))
            
            # Draw deck first (card back)
            self.draw_card(None, 100, y_pos, target_surface)
            
            # Draw available cards
            for i, card in enumerate(level_cards):
                self.draw_card(card, 100 + (i + 1) * (self.CARD_WIDTH + self.CARD_MARGIN), y_pos, target_surface)

        # Draw board coins
        board_coins_text = self.font.render("Board Coins:", True, self.BLACK)
        target_surface.blit(board_coins_text, (1000, BOARD_COINS_Y - 50))
        self.draw_coins(self.game.board.coins, 1000, BOARD_COINS_Y, target_surface)

        # Draw player 1 development gems and coins
        player1_gems_text = self.font.render("Player 1 Development Gems:", True, self.BLACK)
        target_surface.blit(player1_gems_text, (100, PLAYER_CARDS_Y - 25))
        
        # Draw P1 development gem counts with larger squares and spacing
        gem_colors = ['WHITE', 'BLUE', 'GREEN', 'RED', 'BLACK']
        square_size = 40  # Increased from 25
        spacing = 100     # Increased from 80
        for i, (color, count) in enumerate(zip(gem_colors, self.game.players[0].sum_development_card_gem)):
            x = 100 + i * spacing
            y = PLAYER_CARDS_Y + 20
            self.draw_square_gem(color, x, y, square_size, target_surface)
            # Draw count with white text for dark backgrounds, black for light
            text_color = self.WHITE if color in ['BLACK', 'BLUE'] else self.BLACK
            count_text = self.font.render(str(count), True, text_color)
            count_rect = count_text.get_rect(center=(x, y))  # Center in the square
            target_surface.blit(count_text, count_rect)

        # Draw P1 coins
        player1_coins_text = self.font.render("Player 1 Coins:", True, self.BLACK)
        coins_x = 100 + len(gem_colors) * spacing + 60  # Adjusted for new spacing
        target_surface.blit(player1_coins_text, (coins_x, PLAYER_CARDS_Y - 25))
        self.draw_coins(self.game.players[0].coins, coins_x, PLAYER_CARDS_Y + 20, target_surface)

        # Draw player 2 development gems and coins (below player 1)
        player2_y = PLAYER_CARDS_Y + 100
        player2_gems_text = self.font.render("Player 2 Development Gems:", True, self.BLACK)
        target_surface.blit(player2_gems_text, (100, player2_y - 25))
        
        # Draw P2 development gem counts with larger squares and spacing
        for i, (color, count) in enumerate(zip(gem_colors, self.game.players[1].sum_development_card_gem)):
            x = 100 + i * spacing
            y = player2_y + 20
            self.draw_square_gem(color, x, y, square_size, target_surface)
            # Draw count with white text for dark backgrounds, black for light
            text_color = self.WHITE if color in ['BLACK', 'BLUE'] else self.BLACK
            count_text = self.font.render(str(count), True, text_color)
            count_rect = count_text.get_rect(center=(x, y))  # Center in the square
            target_surface.blit(count_text, count_rect)

        # Draw P2 coins
        player2_coins_text = self.font.render("Player 2 Coins:", True, self.BLACK)
        target_surface.blit(player2_coins_text, (coins_x, player2_y - 25))
        self.draw_coins(self.game.players[1].coins, coins_x, player2_y + 20, target_surface)

        pygame.display.flip()

    def run(self):
        running = True
        clock = pygame.time.Clock()
        game_state_changed = True
        
        while running:
            time_delta = clock.tick(60)/1000.0
            current_time = pygame.time.get_ticks()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.ai_button_rect.collidepoint(event.pos):
                        if self.action_state == 'ready':
                            # First click: show probabilities
                            self.show_action_probabilities()
                            self.action_state = 'showing_probs'
                            game_state_changed = True
                        elif self.action_state == 'showing_probs':
                            # Second click: execute action immediately
                            self.take_ai_action()
                            self.action_state = 'ready'
                            game_state_changed = True
                    elif self.auto_play_rect.collidepoint(event.pos):
                        self.is_auto_playing = not self.is_auto_playing
                        self.last_action_time = current_time
                        game_state_changed = True
            
            # Handle auto-play
            if self.is_auto_playing and current_time - self.last_action_time >= self.turn_delay:
                self.show_action_probabilities()  # Show probabilities before action
                self.take_ai_action()
                self.last_action_time = current_time
                game_state_changed = True
            
            if game_state_changed:
                self.screen.fill((200, 200, 200))
                self.draw_game_state()
                game_state_changed = False
            
            pygame.display.flip()

        pygame.quit()

    def draw(self, surface=None):
        # First draw everything from parent class
        super().draw(surface)
        target_surface = surface if surface is not None else self.screen
        
        # Draw AI action button with current state
        button_color = self.PLAYER1_COLOR if self.current_player == 1 else self.PLAYER2_COLOR
        if self.action_requested:
            # Dim the button color when action is pending
            button_color = tuple(max(0, c - 50) for c in button_color)
        pygame.draw.rect(target_surface, button_color, self.ai_button_rect)
        pygame.draw.rect(target_surface, (50, 100, 200), self.ai_button_rect, 2)
        
        # Change button text based on state
        if self.action_state == 'ready':
            text = f"P{self.current_player} Show Probs"
        elif self.action_state == 'showing_probs':
            text = f"P{self.current_player} Execute"
        else:  # pending_action
            text = "Pending..."
            
        text_surface = self.font.render(text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.ai_button_rect.center)
        target_surface.blit(text_surface, text_rect)
        
        # Draw auto-play button
        button_color = (150, 255, 150) if self.is_auto_playing else (100, 150, 255)
        pygame.draw.rect(target_surface, button_color, self.auto_play_rect)
        pygame.draw.rect(target_surface, (50, 100, 200), self.auto_play_rect, 2)
        text_surface = self.font.render("Auto Play", True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.auto_play_rect.center)
        target_surface.blit(text_surface, text_rect)

class AIGameGUI(SplendorGUI):
    def __init__(self, model1, model2, logger, verbose_dict, env):
        # Use environment's game instance
        game = env.env.game  # Get game from SelfPlayEnv
        self.env = env
        
        super().__init__(game)
        
        # Store models and logger
        self.model1 = model1
        self.model2 = model2
        self.logger = logger
        
        # Store verbose settings
        self.verbose_dict = verbose_dict
        self.verbose = verbose_dict['env']
        
        self.current_player = 1  # Track whose turn it is
        
        # Initialize observation
        self.obs = self.env.reset()[0]  # Get initial observation
        
        # Add state tracking
        self.action_state = 'ready'  # 'ready', 'showing_probs', 'pending_action'
        self.action_requested = False
        self.action_request_time = 0
        self.action_delay = 1000  # 1 second delay before action execution
        self.selected_action = None  # Store the selected action
        
        # Create AI action button
        self.ai_button_rect = pygame.Rect(
            self.WINDOW_WIDTH - 280,
            self.WINDOW_HEIGHT - 100,
            120,
            40
        )
        
        # Create auto-play button
        self.auto_play_rect = pygame.Rect(
            self.WINDOW_WIDTH - 150,
            self.WINDOW_HEIGHT - 100,
            120,
            40
        )
        
        # Auto-play settings
        self.is_auto_playing = False
        self.turn_delay = 1000  # Default 1 second delay between auto-play actions
        self.last_action_time = pygame.time.get_ticks()

    def update_observation(self):
        """Get current observation for the active player"""
        player_idx = self.current_player - 1
        if player_idx == 0:
            # For player 1, get observation from environment without resetting
            if hasattr(self, 'last_obs'):
                self.obs = self.last_obs
            else:
                # Only reset on first observation
                self.obs = self.env.reset()[0]
        else:
            # For player 2, get observation directly from game
            self.obs = get_observation(self.game, player_idx, verbose=self.verbose_dict['env'], logger=self.logger)
            action_mask = self.game.players[player_idx].get_all_possible_actions(self.game.board)
            self.obs = np.concatenate([self.obs, action_mask])

    def get_action_probabilities(self, model, obs):
        """Calculate action probabilities for the current state"""
        obs_tensor = torch.FloatTensor(obs)
        
        # Split observation and action mask
        action_mask = obs_tensor[-self.game.players[0].num_actions:]
        obs_features = obs_tensor[:-self.game.players[0].num_actions]
        
        # Move tensors to the same device as the model
        device = next(model.policy.parameters()).device
        obs_features = obs_features.to(device)
        action_mask = action_mask.to(device)
        
        # Create observation dictionary
        obs_dict = {
            'obs': obs_features.unsqueeze(0),
            'action_mask': action_mask.unsqueeze(0)
        }
        
        with torch.no_grad():
            # Get raw logits from policy network
            features = model.policy.extract_features(obs_dict['obs'])
            latent_pi, _ = model.policy.mlp_extractor(features)
            logits = model.policy.action_net(latent_pi)
            
            # Apply action mask
            logits = torch.where(
                obs_dict['action_mask'].bool(),
                logits,
                torch.tensor(-float('inf')).to(logits.device)  # Use -inf instead of -1e+8
            )
            
            # Convert to probabilities
            action_probs = torch.softmax(logits, dim=-1)
            
            # Ensure normalization
            action_probs = action_probs / action_probs.sum()
            
            return action_probs.squeeze(0).cpu().numpy()

    def get_action_description(self, action_idx):
        """Get description for each action based on Player class implementation"""
        if action_idx < 10:  # First 10 actions (0-9) are for three different coins
            candidated_ids = get_coin_comb(action_idx)
            colors = [Element(idx).name for idx in candidated_ids]
            return f"Take three different coins from: {', '.join(colors)}"
        elif action_idx < 15:  # Actions 10-14 are for taking two same coins
            color = Element(action_idx - 10).name
            return f"Take two {color} coins"
        elif action_idx < 27:  # Actions 15-26 are for buying development cards
            card_idx = action_idx - 15
            level = (card_idx // 4) + 1
            pos = card_idx % 4
            cards = (self.game.board.table_level1 if level == 1 else 
                    self.game.board.table_level2 if level == 2 else 
                    self.game.board.table_level3)
            if pos < len(cards):
                card = cards[pos]
                if card:
                    return f"Buy L{level} card: {card.name} (VP: {card.victory_point}, Gem: {card.gem_color})"
            return f"Buy L{level} card at position {pos} (Invalid - no card)"
        elif action_idx < 30:  # Actions 27-29 are for buying reserved cards
            card_idx = action_idx - 27
            player = self.game.players[self.current_player - 1]
            if card_idx < len(player.reserved_cards):
                card = player.reserved_cards[card_idx]
                return f"Buy reserved card: {card.name} (VP: {card.victory_point}, Gem: {card.gem_color})"
            return f"Buy reserved card at position {card_idx} (Invalid - no card)"
        elif action_idx < 42:  # Actions 30-41 are for reserving cards
            card_idx = action_idx - 30
            level = (card_idx // 4) + 1
            pos = card_idx % 4
            cards = (self.game.board.table_level1 if level == 1 else 
                    self.game.board.table_level2 if level == 2 else 
                    self.game.board.table_level3)
            if pos < len(cards):
                card = cards[pos]
                if card:
                    return f"Reserve L{level} card: {card.name} (VP: {card.victory_point}, Gem: {card.gem_color})"
            return f"Reserve L{level} card at position {pos} (Invalid - no card)"
        else:
            return "Unknown action"

    def show_action_probabilities(self):
        """Show action probabilities without executing action"""
        current_model = self.model1 if self.current_player == 1 else self.model2
        current_player = self.game.players[self.current_player - 1]
        
        # Log basic game state
        self.logger.info("\n" + "="*50)
        self.logger.info(f"Turn {current_player.step} - Player {self.current_player}")
        self.logger.info(f"Victory Points - P1: {self.game.players[0].sum_victory_point}, P2: {self.game.players[1].sum_victory_point}")
        
        # Get valid actions
        valid_actions = np.where(self.game.players[self.current_player-1].get_all_possible_actions(self.game.board))[0]
        
        # Show action probabilities
        self.logger.info("\nAll valid action probabilities:")
        total_prob = 0
        
        if isinstance(current_model, RandomPlayer):
            # For random player, distribute probability equally among valid actions
            prob = 1.0 / len(valid_actions)
            for action_idx in valid_actions:
                description = self.get_action_description(action_idx)
                prob_percent = prob * 100
                self.logger.info(f"Action {action_idx:2d} ({prob_percent:5.1f}%): {description}")
                total_prob += prob
        else:
            # For PPO model, get action probabilities
            action_probs = self.get_action_probabilities(current_model, self.obs)
            valid_probs = [(i, action_probs[i]) for i in valid_actions]
            sorted_actions = sorted(valid_probs, key=lambda x: x[0])
            
            for action_idx, prob in sorted_actions:
                description = self.get_action_description(action_idx)
                prob_percent = prob * 100
                self.logger.info(f"Action {action_idx:2d} ({prob_percent:5.1f}%): {description}")
                total_prob += prob
        
        total_percent = total_prob * 100
        self.logger.info(f"\nTotal probability: {total_percent:.1f}%")

        # Print current game state
        if self.verbose:
            self.logger.info(f"\nBoard state:")
            if self.verbose_dict['board']:
                self.game.board.print_status()
            self.logger.info(f"Player state:")
            if self.verbose_dict['player']:
                current_player.print_status()

    def take_ai_action(self):
        # Get current model and player
        current_model = self.model1 if self.current_player == 1 else self.model2
        
        # Get action from current model
        if isinstance(current_model, RandomPlayer):
            action = current_model(self.obs)
        else:
            action, _state = current_model.predict(self.obs, deterministic=False)
        
        # Log action
        self.logger.info(f"\nPlayer {self.current_player} taking action: {action}")
        ActionLogger.log_selected_action(self.logger, action, self.verbose)
        
        # Always use environment step for both players
        self.obs, reward, done, _, info = self.env.step(action)
        self.last_obs = self.obs  # Store observation for next update
        
        # Always log result
        self.logger.info(f"Action result: {reward}")
        
        # Detailed result if verbose
        if self.verbose and self.verbose_dict['player']:
            self.logger.info("Updated player state:")
            self.game.players[self.current_player - 1].print_status()
        
        # Switch current player
        self.current_player = 2 if self.current_player == 1 else 1
        
        # Update observation for next player
        self.update_observation()
        
        # Check for game end
        p1_vp = self.game.players[0].sum_victory_point
        p2_vp = self.game.players[1].sum_victory_point
        if not done and (p1_vp >= 15 or p2_vp >= 15):
            done = True
            
        if done:
            self.is_auto_playing = False
            winner = "Player 1" if p1_vp > p2_vp else "Player 2"
            self.logger.info(f"Game Won by {winner}! Starting new episode...")
            if self.verbose:
                self.logger.info("Final game state:")
                if self.verbose_dict['game']:
                    self.game.print_status()
            
            # Reset game through environment
            self.obs = self.env.reset()[0]
            self.current_player = 1
        
        return done

    def run(self):
        running = True
        clock = pygame.time.Clock()
        game_state_changed = True
        
        while running:
            time_delta = clock.tick(60)/1000.0
            current_time = pygame.time.get_ticks()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.ai_button_rect.collidepoint(event.pos):
                        if self.action_state == 'ready':
                            # First click: show probabilities
                            self.show_action_probabilities()
                            self.action_state = 'showing_probs'
                            game_state_changed = True
                        elif self.action_state == 'showing_probs':
                            # Second click: execute action immediately
                            self.take_ai_action()
                            self.action_state = 'ready'
                            game_state_changed = True
                    elif self.auto_play_rect.collidepoint(event.pos):
                        self.is_auto_playing = not self.is_auto_playing
                        self.last_action_time = current_time
                        game_state_changed = True
            
            # Handle auto-play
            if self.is_auto_playing and current_time - self.last_action_time >= self.turn_delay:
                self.show_action_probabilities()  # Show probabilities before action
                self.take_ai_action()
                self.last_action_time = current_time
                game_state_changed = True
            
            if game_state_changed:
                self.draw()
                game_state_changed = False
            
            pygame.display.flip()

        pygame.quit()

    def draw(self, surface=None):
        """Draw the complete GUI including game state and AI controls"""
        target_surface = surface if surface is not None else self.screen
        
        # Draw base game state
        self.draw_game_state(target_surface)
        
        # Draw AI-specific elements
        self.draw_ai_buttons(target_surface)

    def draw_ai_buttons(self, surface):
        """Draw AI control buttons"""
        # Draw AI action button
        button_color = self.PLAYER1_COLOR if self.current_player == 1 else self.PLAYER2_COLOR
        pygame.draw.rect(surface, button_color, self.ai_button_rect)
        pygame.draw.rect(surface, (50, 100, 200), self.ai_button_rect, 2)
        
        # Draw button text
        text = f"P{self.current_player} Execute" if self.action_state == 'showing_probs' else f"P{self.current_player} Show Probs"
        text_surface = self.font.render(text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.ai_button_rect.center)
        surface.blit(text_surface, text_rect)
        
        # Draw auto-play button
        button_color = (150, 255, 150) if self.is_auto_playing else (100, 150, 255)
        pygame.draw.rect(surface, button_color, self.auto_play_rect)
        pygame.draw.rect(surface, (50, 100, 200), self.auto_play_rect, 2)
        text_surface = self.font.render("Auto Play", True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.auto_play_rect.center)
        surface.blit(text_surface, text_rect) 