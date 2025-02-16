import pygame
import os
import numpy as np
from jsplendor.env import JsplendorEnv
from jsplendor.utils import TestLogger, Element
from stable_baselines3 import PPO
from jsplendor.env.observation import get_observation
import torch
from jsplendor.game.utils import get_coin_comb

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

    def draw(self, surface=None):
        target_surface = surface if surface is not None else self.screen
        
        # Fill background
        target_surface.fill((200, 200, 200))

        # Draw step counter, victory points, and noble count at top center
        step_text = self.font.render(f"Step: {self.game.step}", True, self.BLACK)
        vp_text = self.font.render(f"Victory Points: {self.game.player1.sum_victory_point}", True, self.BLACK)
        noble_text = self.font.render(f"Nobles: {len(self.game.player1.noble_cards)}", True, self.BLACK)
        
        # Position step counter, VP, and noble count with spacing
        step_rect = step_text.get_rect(center=(self.WINDOW_WIDTH // 2 - 200, 30))
        vp_rect = vp_text.get_rect(center=(self.WINDOW_WIDTH // 2, 30))
        noble_rect = noble_text.get_rect(center=(self.WINDOW_WIDTH // 2 + 200, 30))
        
        target_surface.blit(step_text, step_rect)
        target_surface.blit(vp_text, vp_rect)
        target_surface.blit(noble_text, noble_rect)

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

        # Draw player development gems and coins in one row
        player_gems_text = self.font.render("Player Development Gems:", True, self.BLACK)
        target_surface.blit(player_gems_text, (100, PLAYER_CARDS_Y - 25))
        
        # Draw development gem counts
        gem_colors = ['WHITE', 'BLUE', 'GREEN', 'RED', 'BLACK']
        for i, (color, count) in enumerate(zip(gem_colors, self.game.player1.sum_development_card_gem)):
            # Draw square gem icon
            self.draw_square_gem(color, 100 + i * 80, PLAYER_CARDS_Y + 20, 25, target_surface)
            # Draw count
            count_text = self.font.render(str(count), True, self.BLACK)
            count_rect = count_text.get_rect(center=(100 + i * 80, PLAYER_CARDS_Y + 50))
            target_surface.blit(count_text, count_rect)

        # Draw player coins next to development gems
        player_coins_text = self.font.render("Player Coins:", True, self.BLACK)
        coins_x = 100 + len(gem_colors) * 80 + 60  # Position after gems with some spacing
        target_surface.blit(player_coins_text, (coins_x, PLAYER_CARDS_Y - 25))
        self.draw_coins(self.game.player1.coins, coins_x, PLAYER_CARDS_Y + 20, target_surface)

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
                        # Show probabilities before taking action
                        self.get_action_probabilities()
                        self.take_ai_action()
                        game_state_changed = True
                    elif self.auto_play_rect.collidepoint(event.pos):
                        self.is_auto_playing = not self.is_auto_playing
                        self.last_action_time = current_time
                        game_state_changed = True
                    elif self.speed_up_rect.collidepoint(event.pos):
                        self.turn_delay = max(100, self.turn_delay - 100)  # Minimum 0.1s
                        game_state_changed = True
                    elif self.speed_down_rect.collidepoint(event.pos):
                        self.turn_delay = min(5000, self.turn_delay + 100)  # Maximum 5s
                        game_state_changed = True
            
            if self.is_auto_playing and current_time - self.last_action_time >= self.turn_delay:
                # Show probabilities before taking action
                self.get_action_probabilities()
                done = self.take_ai_action()
                self.last_action_time = current_time
                game_state_changed = True
            
            if game_state_changed:
                self.screen.fill((200, 200, 200))
                self.draw()
                game_state_changed = False
            
            pygame.display.flip()

        pygame.quit()

class AIGameGUI(SplendorGUI):
    def __init__(self, game, env, model, logger):
        super().__init__(game)
        
        # Store environment and model
        self.env = env
        self.model = model
        self.logger = logger
        self.verbose = True
        
        # Initialize observation without resetting the env
        self.obs = get_observation(self.env.game)
        action_mask = self.env.get_action_mask()
        self.obs = np.concatenate([self.obs, action_mask])
        
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
        self.turn_delay = 1000  # Default 1 second delay (in milliseconds)
        self.last_action_time = pygame.time.get_ticks()
        
        # Create speed control buttons with delay display in between
        button_size = 40
        spacing = 40
        total_width = button_size * 2 + spacing
        base_x = self.WINDOW_WIDTH - 410
        
        self.speed_down_rect = pygame.Rect(
            base_x,
            self.WINDOW_HEIGHT - 100,
            button_size,
            button_size
        )
        
        self.speed_up_rect = pygame.Rect(
            base_x + button_size + spacing,
            self.WINDOW_HEIGHT - 100,
            button_size,
            button_size
        )
        
        # Sync initial state
        self.sync_game_state()

    def sync_game_state(self):
        # Sync board state
        self.game.board.coins = self.env.game.board.coins.copy()
        self.game.board.noble_cards = self.env.game.board.noble_cards.copy()
        
        # Sync table cards
        self.game.board.table_level1 = self.env.game.board.table_level1.copy()
        self.game.board.table_level2 = self.env.game.board.table_level2.copy()
        self.game.board.table_level3 = self.env.game.board.table_level3.copy()
        
        # Sync deck cards
        self.game.board.level1_cards = self.env.game.board.level1_cards.copy()
        self.game.board.level2_cards = self.env.game.board.level2_cards.copy()
        self.game.board.level3_cards = self.env.game.board.level3_cards.copy()
        
        # Update flattened table cards
        self.game.board._flatten_table_cards()
        
        # Sync player state
        self.game.player1.coins = self.env.game.player1.coins.copy()
        self.game.player1.development_cards = self.env.game.player1.development_cards.copy()
        self.game.player1.noble_cards = self.env.game.player1.noble_cards.copy()
        
        # Update player's score and gem counts
        self.game.player1._update_score()
        
        # Sync game step
        self.game.step = self.env.game.step
        
        # Calculate and show probabilities for the new state
        self.get_action_probabilities()

    def get_action_probabilities(self):
        """Calculate and log current action probabilities"""
        obs_tensor = torch.FloatTensor(self.obs)
        
        # Split observation and action mask
        action_mask = obs_tensor[-self.env.action_space.n:]
        obs_features = obs_tensor[:-self.env.action_space.n]
        
        # Move tensors to the same device as the model
        device = next(self.model.policy.parameters()).device
        obs_features = obs_features.to(device)
        action_mask = action_mask.to(device)
        
        # Create observation dictionary
        obs_dict = {
            'obs': obs_features.unsqueeze(0),
            'action_mask': action_mask.unsqueeze(0)
        }
        
        with torch.no_grad():
            # Get raw logits from policy network
            features = self.model.policy.extract_features(obs_dict['obs'])
            latent_pi, _ = self.model.policy.mlp_extractor(features)
            logits = self.model.policy.action_net(latent_pi)
            
            # Apply action mask
            logits = torch.where(
                obs_dict['action_mask'].bool(),
                logits,
                torch.tensor(-1e+8).to(logits.device)
            )
            
            # Convert to probabilities
            action_probs = torch.softmax(logits, dim=-1)
            action_probs = action_probs.squeeze(0).cpu().numpy()
            
            # Log probabilities
            if self.verbose:
                valid_actions = np.where(self.env.get_action_mask())[0]
                self.logger.info("-" * 30)
                self.logger.info("Available actions:")
                for valid_action in valid_actions:
                    prob = action_probs[valid_action] * 100
                    if prob > 0.1:  # Only show significant probabilities
                        if valid_action < 10:
                            # Get coin combination for this action using the imported function
                            coin_ids = get_coin_comb(valid_action)
                            coins = [Element(ids).name for ids in coin_ids]
                            desc = f"Get coins: {', '.join(coins)}"
                        else:
                            card_pos = valid_action - 10
                            card = self.env.game.board.flatten_table_cards[card_pos]
                            if card:
                                desc = f"Buy {card.name} (Level: {card.level}, VP: {card.victory_point}, Color: {card.gem_color})"
                            else:
                                desc = "Buy card (empty slot)"
                        self.logger.info(f"  Action {valid_action}: {desc} ({prob:.1f}%)")

    def take_ai_action(self):
        # Get action from model
        action, _state = self.model.predict(self.obs, deterministic=False)
        
        # Log only the selected action
        if self.verbose:
            self.logger.info(f"Selected: Action {action}")
        
        # Take step in environment
        self.obs, reward, done, _, info = self.env.step(action)
        
        # Store reward in game object for display
        self.game.last_reward = reward
        
        # Sync game state with environment (this will also show new probabilities)
        self.sync_game_state()
        
        if done:
            self.is_auto_playing = False
            if self.env.game.player1.sum_victory_point >= self.env.target_vp:
                self.logger.info("Game Won! Starting new episode...")
            else:
                self.logger.info("Episode ended (max steps reached)")
            self.obs, _ = self.env.reset()
            self.game.reset()
            self.sync_game_state()
            self.game.last_reward = 0.0
        
        return done

    def draw(self, surface=None):
        # First draw everything from parent class
        super().draw(surface)
        target_surface = surface if surface is not None else self.screen
        
        # Draw AI action button
        pygame.draw.rect(target_surface, (100, 150, 255), self.ai_button_rect)
        pygame.draw.rect(target_surface, (50, 100, 200), self.ai_button_rect, 2)
        text_surface = self.font.render("AI Action", True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.ai_button_rect.center)
        target_surface.blit(text_surface, text_rect)
        
        # Draw auto-play button with different color when active
        button_color = (150, 255, 150) if self.is_auto_playing else (100, 150, 255)
        pygame.draw.rect(target_surface, button_color, self.auto_play_rect)
        pygame.draw.rect(target_surface, (50, 100, 200), self.auto_play_rect, 2)
        text_surface = self.font.render("Auto Play", True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.auto_play_rect.center)
        target_surface.blit(text_surface, text_rect)
        
        # Draw speed control buttons and delay display
        # Draw "-" button
        pygame.draw.rect(target_surface, (100, 150, 255), self.speed_down_rect)
        pygame.draw.rect(target_surface, (50, 100, 200), self.speed_down_rect, 2)
        text_surface = self.font.render("-", True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.speed_down_rect.center)
        target_surface.blit(text_surface, text_rect)
        
        # Draw "+" button
        pygame.draw.rect(target_surface, (100, 150, 255), self.speed_up_rect)
        pygame.draw.rect(target_surface, (50, 100, 200), self.speed_up_rect, 2)
        text_surface = self.font.render("+", True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.speed_up_rect.center)
        target_surface.blit(text_surface, text_rect)
        
        # Draw delay value centered between buttons
        delay_text = self.font.render(f"{self.turn_delay/1000:.1f}s", True, (0, 0, 0))
        center_x = (self.speed_down_rect.centerx + self.speed_up_rect.centerx) // 2
        delay_rect = delay_text.get_rect(center=(center_x, self.speed_down_rect.centery))
        target_surface.blit(delay_text, delay_rect) 