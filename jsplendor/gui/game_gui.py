import pygame
import os
import numpy as np
import torch
from jsplendor.utils import TestLogger, Element
from jsplendor.models.random_player import RandomPlayer
from jsplendor.utils.config import get_coin_comb

class GameGUI:
    def __init__(self, model1, model2, logger, verbose_dict, env):
        pygame.init()
        
        # Store environment and game instance
        self.env = env
        self.game = env.env.game
        
        # Store models and logger
        self.model1 = model1
        self.model2 = model2
        self.logger = logger
        
        # Store verbose settings
        self.verbose_dict = verbose_dict
        self.verbose = verbose_dict['env']
        
        # Window settings
        self.WINDOW_WIDTH = 1600
        self.WINDOW_HEIGHT = 1200
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("JSplendor")

        # Colors
        self.COLORS = {
            'WHITE': (240, 240, 240),
            'BLUE': (0, 0, 255),
            'GREEN': (0, 255, 0),
            'RED': (255, 0, 0),
            'BLACK': (0, 0, 0),
            'GOLD': (255, 215, 0)
        }

        # Gem borders
        self.GEM_BORDERS = {
            'WHITE': (128, 128, 128),
            'BLUE': (0, 0, 180),
            'GREEN': (0, 180, 0),
            'RED': (180, 0, 0),
            'BLACK': (64, 64, 64),
            'GOLD': (180, 150, 0)
        }

        # Card settings
        self.CARD_WIDTH = 150
        self.CARD_HEIGHT = 200
        self.CARD_MARGIN = 20
        self.CARD_COLORS = {
            1: (255, 200, 200),  # Level 1: pink
            2: (200, 255, 200),  # Level 2: mint green
            3: (200, 200, 255),  # Level 3: light blue
        }

        # Initialize font
        self.font = pygame.font.Font(None, 24)
        
        # Game state tracking
        self.current_player = 1
        self.update_observation()
        
        # Button settings
        self.step_button_rect = pygame.Rect(
            self.WINDOW_WIDTH - 280,
            self.WINDOW_HEIGHT - 100,
            120,
            40
        )
        
        self.auto_play_rect = pygame.Rect(
            self.WINDOW_WIDTH - 150,
            self.WINDOW_HEIGHT - 100,
            120,
            40
        )
        
        # Action state tracking
        self.action_state = 'ready'  # 'ready', 'showing_probs'
        self.is_auto_playing = False
        self.turn_delay = 1000  # 1 second delay between auto-play actions
        self.last_action_time = pygame.time.get_ticks()

    def update_observation(self):
        """Get current observation from environment"""
        if hasattr(self, 'last_obs'):
            self.obs = self.last_obs
        else:
            self.obs = self.env.reset()[0]

    def get_action_probabilities(self, model, obs):
        """Calculate action probabilities for current state"""
        obs_tensor = torch.FloatTensor(obs)
        
        action_mask = obs_tensor[-self.game.players[0].num_actions:]
        obs_features = obs_tensor[:-self.game.players[0].num_actions]
        
        device = next(model.policy.parameters()).device
        obs_features = obs_features.to(device)
        action_mask = action_mask.to(device)
        
        obs_dict = {
            'obs': obs_features.unsqueeze(0),
            'action_mask': action_mask.unsqueeze(0)
        }
        
        with torch.no_grad():
            features = model.policy.extract_features(obs_dict['obs'])
            latent_pi, _ = model.policy.mlp_extractor(features)
            logits = model.policy.action_net(latent_pi)
            
            logits = torch.where(
                obs_dict['action_mask'].bool(),
                logits,
                torch.tensor(-float('inf')).to(logits.device)
            )
            
            action_probs = torch.softmax(logits, dim=-1)
            action_probs = action_probs / action_probs.sum()
            
            return action_probs.squeeze(0).cpu().numpy()

    def get_action_description(self, action_idx):
        """Get human-readable description of an action"""
        if action_idx < 10:  # Take three different coins
            candidated_ids = get_coin_comb(action_idx)
            colors = [Element(idx).name for idx in candidated_ids]
            return f"Take three different coins from: {', '.join(colors)}"
        elif action_idx < 15:  # Take two same coins
            color = Element(action_idx - 10).name
            return f"Take two {color} coins"
        elif action_idx < 27:  # Buy development card
            card_idx = action_idx - 15
            level = (card_idx // 4) + 1
            pos = card_idx % 4
            cards = (self.game.board.table_level1 if level == 1 else 
                    self.game.board.table_level2 if level == 2 else 
                    self.game.board.table_level3)
            if pos < len(cards) and cards[pos]:
                card = cards[pos]
                return f"Buy L{level} card: {card.name} (VP: {card.victory_point}, Gem: {card.gem_color})"
            return f"Buy L{level} card at position {pos} (Invalid - no card)"
        elif action_idx < 30:  # Buy reserved card
            card_idx = action_idx - 27
            player = self.game.players[self.current_player - 1]
            if card_idx < len(player.reserved_cards):
                card = player.reserved_cards[card_idx]
                return f"Buy reserved card: {card.name} (VP: {card.victory_point}, Gem: {card.gem_color})"
            return f"Buy reserved card at position {card_idx} (Invalid - no card)"
        elif action_idx < 42:  # Reserve card
            card_idx = action_idx - 30
            level = (card_idx // 4) + 1
            pos = card_idx % 4
            cards = (self.game.board.table_level1 if level == 1 else 
                    self.game.board.table_level2 if level == 2 else 
                    self.game.board.table_level3)
            if pos < len(cards) and cards[pos]:
                card = cards[pos]
                return f"Reserve L{level} card: {card.name} (VP: {card.victory_point}, Gem: {card.gem_color})"
            return f"Reserve L{level} card at position {pos} (Invalid - no card)"
        return "Unknown action"

    def show_action_probabilities(self):
        """Show action probabilities for current player"""
        current_model = self.model1 if self.current_player == 1 else self.model2
        current_player = self.game.players[self.current_player - 1]
        
        self.logger.info("\n" + "="*50)
        self.logger.info(f"Turn {current_player.step} - Player {self.current_player}")
        self.logger.info(f"Victory Points - P1: {self.game.players[0].sum_victory_point}, P2: {self.game.players[1].sum_victory_point}")
        
        valid_actions = np.where(self.game.players[self.current_player-1].get_all_possible_actions(self.game.board))[0]
        
        self.logger.info("\nAll valid action probabilities:")
        total_prob = 0
        
        if isinstance(current_model, RandomPlayer):
            prob = 1.0 / len(valid_actions)
            for action_idx in valid_actions:
                description = self.get_action_description(action_idx)
                prob_percent = prob * 100
                self.logger.info(f"Action {action_idx:2d} ({prob_percent:5.1f}%): {description}")
                total_prob += prob
        else:
            action_probs = self.get_action_probabilities(current_model, self.obs)
            valid_probs = [(i, action_probs[i]) for i in valid_actions]
            sorted_actions = sorted(valid_probs, key=lambda x: x[1], reverse=True)
            
            for action_idx, prob in sorted_actions:
                description = self.get_action_description(action_idx)
                prob_percent = prob * 100
                self.logger.info(f"Action {action_idx:2d} ({prob_percent:5.1f}%): {description}")
                total_prob += prob
        
        self.logger.info(f"\nTotal probability: {total_prob*100:.1f}%")

        if self.verbose:
            self.logger.info("\nBoard state:")
            if self.verbose_dict['board']:
                self.game.board.print_status()
            self.logger.info("Player state:")
            if self.verbose_dict['player']:
                current_player.print_status()

    def take_step(self):
        """Execute a full step (show probabilities and take action)"""
        self.show_action_probabilities()
        self.take_ai_action()
        self.action_state = 'ready'

    def take_ai_action(self):
        """Execute AI action for current player"""
        current_model = self.model1 if self.current_player == 1 else self.model2
        
        action = (current_model(self.obs) if isinstance(current_model, RandomPlayer) 
                 else current_model.predict(self.obs, deterministic=False)[0])
        
        self.logger.info(f"\nPlayer {self.current_player} taking action: {action}")
        
        self.obs, reward, done, _, info = self.env.step(action)
        self.last_obs = self.obs
        
        self.logger.info(f"Action result: {reward}")
        
        if self.verbose and self.verbose_dict['player']:
            self.logger.info("Updated player state:")
            self.game.players[self.current_player - 1].print_status()
        
        self.current_player = 2 if self.current_player == 1 else 1
        self.update_observation()
        
        p1_vp = self.game.players[0].sum_victory_point
        p2_vp = self.game.players[1].sum_victory_point
        if not done and (p1_vp >= 15 or p2_vp >= 15):
            done = True
            
        if done:
            self.is_auto_playing = False
            winner = "Player 1" if p1_vp > p2_vp else "Player 2"
            self.logger.info(f"Game Won by {winner}! Starting new episode...")
            if self.verbose and self.verbose_dict['game']:
                self.logger.info("Final game state:")
                self.game.print_status()
            
            self.obs = self.env.reset()[0]
            self.current_player = 1
        
        return done

    def draw_gem(self, color, x, y, radius, surface=None):
        """Draw a gem with border"""
        target_surface = surface if surface is not None else self.screen
        pygame.draw.circle(target_surface, self.GEM_BORDERS[color], (x, y), radius)
        pygame.draw.circle(target_surface, self.COLORS[color], (x, y), radius - 2)

    def draw_card(self, card, x, y, surface=None):
        """Draw a card with all its information"""
        target_surface = surface if surface is not None else self.screen
        
        if card is None:
            pygame.draw.rect(target_surface, (139, 69, 19), (x, y, self.CARD_WIDTH, self.CARD_HEIGHT))
            pygame.draw.rect(target_surface, (101, 67, 33), (x, y, self.CARD_WIDTH, self.CARD_HEIGHT), 2)
            pygame.draw.rect(target_surface, (101, 67, 33), 
                           (x + 10, y + 10, self.CARD_WIDTH - 20, self.CARD_HEIGHT - 20), 2)
            return

        bg_color = self.CARD_COLORS[card.level]
        pygame.draw.rect(target_surface, bg_color, (x, y, self.CARD_WIDTH, self.CARD_HEIGHT))
        pygame.draw.rect(target_surface, (0, 0, 0), (x, y, self.CARD_WIDTH, self.CARD_HEIGHT), 2)

        name_text = self.font.render(card.name, True, (0, 0, 0))
        target_surface.blit(name_text, (x + 5, y + 5))

        vp_text = self.font.render(f"VP: {card.victory_point}", True, (0, 0, 0))
        target_surface.blit(vp_text, (x + 5, y + 25))

        self.draw_gem(card.gem_color, x + self.CARD_WIDTH - 20, y + 20, 12, target_surface)

        y_offset = 50
        for color, amount in zip(['WHITE', 'BLUE', 'GREEN', 'RED', 'BLACK'], card.price):
            if amount > 0:
                self.draw_gem(color, x + 15, y + y_offset + 8, 8, target_surface)
                price_text = self.font.render(str(amount), True, (0, 0, 0))
                target_surface.blit(price_text, (x + 30, y + y_offset))
                y_offset += 20

    def draw_noble(self, noble, x, y, surface=None):
        """Draw a noble card"""
        if not noble:
            return
            
        target_surface = surface if surface is not None else self.screen
        pygame.draw.rect(target_surface, (255, 223, 196), (x, y, 120, 120))
        pygame.draw.rect(target_surface, (0, 0, 0), (x, y, 120, 120), 2)
        
        noble_text = self.font.render(f"Noble: {noble.victory_point}VP", True, (0, 0, 0))
        target_surface.blit(noble_text, (x + 10, y + 10))
        
        y_offset = 40
        for color, amount in zip(['WHITE', 'BLUE', 'GREEN', 'RED', 'BLACK'], noble.price):
            if amount > 0:
                self.draw_gem(color, x + 25, y + y_offset, 8, target_surface)
                price_text = self.font.render(str(amount), True, (0, 0, 0))
                target_surface.blit(price_text, (x + 45, y + y_offset - 5))
                y_offset += 20

    def draw_coins(self, coins, x, y, surface=None):
        """Draw coin counts"""
        target_surface = surface if surface is not None else self.screen
        for i, (color, count) in enumerate(coins.items()):
            self.draw_gem(color, x + i * 60, y, 25, target_surface)
            text_color = (255, 255, 255) if color in ['BLACK', 'BLUE'] else (0, 0, 0)
            count_text = self.font.render(str(count), True, text_color)
            text_rect = count_text.get_rect(center=(x + i * 60, y))
            target_surface.blit(count_text, text_rect)

    def draw_game_state(self, surface=None):
        """Draw the complete game state"""
        target_surface = surface if surface is not None else self.screen
        target_surface.fill((200, 200, 200))

        # Draw player info
        for p in [0, 1]:
            x = 20 if p == 0 else self.WINDOW_WIDTH - 250
            player = self.game.players[p]
            step_text = self.font.render(f"Player {p+1} - Step: {player.step}", True, (0, 0, 0))
            vp_text = self.font.render(f"Victory Points: {player.sum_victory_point}", True, (0, 0, 0))
            noble_text = self.font.render(f"Nobles: {len(player.noble_cards)}", True, (0, 0, 0))
            target_surface.blit(step_text, (x, 10))
            target_surface.blit(vp_text, (x, 30))
            target_surface.blit(noble_text, (x, 50))

        # Draw nobles
        for i, noble in enumerate(self.game.board.noble_cards):
            self.draw_noble(noble, 100 + i * 160, 80, target_surface)

        # Draw development cards
        levels = [
            (self.game.board.table_level3, "Level 3", 250),
            (self.game.board.table_level2, "Level 2", 500),
            (self.game.board.table_level1, "Level 1", 750),
        ]
        
        for cards, label, y_pos in levels:
            label_text = self.font.render(label, True, (0, 0, 0))
            target_surface.blit(label_text, (100, y_pos - 25))
            
            self.draw_card(None, 100, y_pos, target_surface)
            
            for i, card in enumerate(cards):
                self.draw_card(card, 100 + (i + 1) * (self.CARD_WIDTH + self.CARD_MARGIN), y_pos, target_surface)

        # Draw board coins
        board_coins_text = self.font.render("Board Coins:", True, (0, 0, 0))
        target_surface.blit(board_coins_text, (1000, 250))
        self.draw_coins(self.game.board.coins, 1000, 300, target_surface)

        # Draw player gems and coins
        for p in [0, 1]:
            y_base = 1000 + p * 100
            player = self.game.players[p]
            
            gems_text = self.font.render(f"Player {p+1} Development Gems:", True, (0, 0, 0))
            target_surface.blit(gems_text, (100, y_base - 25))
            
            for i, (color, count) in enumerate(zip(['WHITE', 'BLUE', 'GREEN', 'RED', 'BLACK'], 
                                                 player.sum_development_card_gem)):
                x = 100 + i * 100
                y = y_base + 20
                self.draw_square_gem(color, x, y, 40, target_surface)
                text_color = (255, 255, 255) if color in ['BLACK', 'BLUE'] else (0, 0, 0)
                count_text = self.font.render(str(count), True, text_color)
                count_rect = count_text.get_rect(center=(x, y))
                target_surface.blit(count_text, count_rect)
            
            coins_text = self.font.render(f"Player {p+1} Coins:", True, (0, 0, 0))
            target_surface.blit(coins_text, (700, y_base - 25))
            self.draw_coins(player.coins, 700, y_base + 20, target_surface)

    def draw_square_gem(self, color, x, y, size, surface=None):
        """Draw a square gem with border"""
        target_surface = surface if surface is not None else self.screen
        # Draw border
        pygame.draw.rect(target_surface, self.GEM_BORDERS[color], 
                        (x - size//2, y - size//2, size, size))
        # Draw interior
        pygame.draw.rect(target_surface, self.COLORS[color], 
                        (x - size//2 + 2, y - size//2 + 2, size - 4, size - 4))

    def draw_buttons(self, surface=None):
        """Draw control buttons"""
        target_surface = surface if surface is not None else self.screen
        
        # Draw step button
        pygame.draw.rect(target_surface, (120, 170, 200), self.step_button_rect)
        pygame.draw.rect(target_surface, (50, 100, 200), self.step_button_rect, 2)
        text = "Take Step"
        text_surface = self.font.render(text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.step_button_rect.center)
        target_surface.blit(text_surface, text_rect)
        
        # Draw auto-play button
        button_color = (150, 255, 150) if self.is_auto_playing else (100, 150, 255)
        pygame.draw.rect(target_surface, button_color, self.auto_play_rect)
        pygame.draw.rect(target_surface, (50, 100, 200), self.auto_play_rect, 2)
        text_surface = self.font.render("Auto Play", True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.auto_play_rect.center)
        target_surface.blit(text_surface, text_rect)

    def draw(self, surface=None):
        """Draw complete GUI"""
        target_surface = surface if surface is not None else self.screen
        self.draw_game_state(target_surface)
        self.draw_buttons(target_surface)

    def run(self):
        """Main game loop"""
        running = True
        clock = pygame.time.Clock()
        game_state_changed = True
        
        while running:
            clock.tick(60)
            current_time = pygame.time.get_ticks()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.step_button_rect.collidepoint(event.pos):
                        if self.action_state == 'ready':
                            self.take_step()
                            game_state_changed = True
                    elif self.auto_play_rect.collidepoint(event.pos):
                        self.is_auto_playing = not self.is_auto_playing
                        self.last_action_time = current_time
                        game_state_changed = True
            
            if self.is_auto_playing and current_time - self.last_action_time >= self.turn_delay:
                self.take_step()
                self.last_action_time = current_time
                game_state_changed = True
            
            if game_state_changed:
                self.draw()
                game_state_changed = False
            
            pygame.display.flip()

        pygame.quit() 