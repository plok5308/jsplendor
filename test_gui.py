import pygame
from jsplendor.game import Game
from jsplendor.gui.game_gui import SplendorGUI
from jsplendor.utils import get_verbose_dict, TestLogger
from jsplendor.env import JsplendorEnv
from stable_baselines3 import PPO
import os

class AIControlledGUI:
    def __init__(self):
        # Enable verbose logging for all components
        self.verbose_dict = get_verbose_dict(False)  # Start with all False
        self.verbose_dict['env'] = True
        self.verbose_dict['game'] = True
        self.verbose_dict['board'] = True
        self.verbose_dict['player'] = True
        
        # Initialize environment and load model
        self.env = JsplendorEnv(self.verbose_dict)
        exp = '250215_small_step_penalty'
        model_path = f'logs/{exp}/best_model'
        log_dir = f'logs/{exp}/'
        
        # Initialize logger
        self.logger = TestLogger(log_dir)
        self.logger.info("Starting GUI test session")
        
        # Load model
        self.logger.info("Loading pretrained model...")
        self.model = PPO.load(model_path, env=self.env)
        self.logger.info('Model loaded successfully')
        
        # Initialize observation
        self.obs, _ = self.env.reset()
        
        # Create game instance and sync with environment's initial state
        self.game = Game(self.verbose_dict)
        self.game.last_reward = 0.0
        self.sync_game_state()
        
        # Create GUI with the game instance
        self.gui = SplendorGUI(self.game)
        
        # Create background surface
        self.background = pygame.Surface((self.gui.WINDOW_WIDTH, self.gui.WINDOW_HEIGHT))
        
        # Create AI action button as a regular pygame rect instead of pygame_gui button
        self.ai_button_rect = pygame.Rect(
            self.gui.WINDOW_WIDTH - 150,
            self.gui.WINDOW_HEIGHT - 100,
            120,
            40
        )
        
        # Pass the button rect to the GUI
        self.gui.ai_button_rect = self.ai_button_rect
        
        # Initial draw of the game state
        self.gui.draw(self.background)

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

    def take_ai_action(self):
        # Get action from model
        action, _state = self.model.predict(self.obs, deterministic=False)
        
        # Take step in environment
        self.obs, reward, done, _, info = self.env.step(action)
        
        # Store reward in game object for display
        self.game.last_reward = reward
        
        # Sync game state with environment
        self.sync_game_state()
        
        if done and self.env.game.player1.sum_victory_point >= self.env.target_vp:
            self.obs, _ = self.env.reset()
            self.game.reset()
            self.sync_game_state()
            self.game.last_reward = 0.0

    def run(self):
        running = True
        clock = pygame.time.Clock()
        game_state_changed = True
        
        while running:
            time_delta = clock.tick(60)/1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Check if AI button was clicked
                    if self.ai_button_rect.collidepoint(event.pos):
                        self.take_ai_action()
                        game_state_changed = True
            
            # Only redraw game state if it changed
            if game_state_changed:
                self.background.fill((200, 200, 200))
                self.gui.draw(self.background)
                game_state_changed = False
            
            # Draw everything
            self.gui.screen.blit(self.background, (0, 0))
            
            # Single flip per frame
            pygame.display.flip()

def main():
    app = AIControlledGUI()
    try:
        app.run()
    finally:
        # Ensure we log the end of the session
        app.logger.info("GUI test session ended")

if __name__ == "__main__":
    main() 