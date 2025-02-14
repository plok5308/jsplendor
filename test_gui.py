import pygame
import pygame_gui
from jsplendor.game import Game
from jsplendor.gui.game_gui import SplendorGUI
from jsplendor.utils import get_verbose_dict
from jsplendor.env import JsplendorEnv
from stable_baselines3 import PPO

class AIControlledGUI:
    def __init__(self):
        self.verbose_dict = get_verbose_dict(False)
        
        # Initialize environment and load model
        self.env = JsplendorEnv(self.verbose_dict)
        exp = '250213'
        model_path = f'logs/{exp}/best_model'
        self.model = PPO.load(model_path, env=self.env)
        
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
        
        # Initialize pygame_gui
        self.gui_manager = pygame_gui.UIManager((self.gui.WINDOW_WIDTH, self.gui.WINDOW_HEIGHT))
        
        # Set theme colors
        self.gui_manager.get_theme().load_theme({
            "defaults": {
                "colours": {
                    "normal_bg": "#CCCCCC",
                    "dark_bg": "#999999",
                    "normal_text": "#000000",
                    "normal_border": "#999999",
                }
            }
        })
        
        # Create text box for logging
        self.log_box = pygame_gui.elements.UITextBox(
            html_text="Game Started\n",
            relative_rect=pygame.Rect(
                (self.gui.WINDOW_WIDTH - 350, self.gui.WINDOW_HEIGHT - 400),
                (300, 280)
            ),
            manager=self.gui_manager
        )
        
        # Create AI action button
        self.button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                (self.gui.WINDOW_WIDTH - 150, self.gui.WINDOW_HEIGHT - 100),
                (120, 40)
            ),
            text="AI Action",
            manager=self.gui_manager
        )
        
        # Initial draw of the game state
        self.gui.draw(self.background)

    def update_log(self, message):
        current_text = self.log_box.html_text
        self.log_box.html_text = current_text + message + "\n"
        self.log_box.rebuild()

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
        
        # Store previous state for comparison
        prev_dev_cards = set(self.env.game.player1.development_cards.copy())
        prev_noble_cards = set(self.env.game.player1.noble_cards.copy())
        
        # Take step in environment
        self.obs, reward, done, _, info = self.env.step(action)
        
        # Debug print
        print(f"Step result: reward={reward}, done={done}, victory_points={self.env.game.player1.sum_victory_point}")
        
        # Store reward in game object for display
        self.game.last_reward = reward
        
        # Log card acquisition after action
        if info.get('is_get_card'):
            # Compare with previous state to find new card
            current_dev_cards = set(self.env.game.player1.development_cards)
            new_cards = current_dev_cards - prev_dev_cards
            if new_cards:
                new_card = list(new_cards)[0]
                card_info = (
                    f"Got {new_card.name} card:\n"
                    f"  Level: {new_card.level}\n"
                    f"  VP: {new_card.victory_point}\n"
                    f"  Color: {new_card.gem_color}"
                )
                self.update_log(card_info)
        
        # Sync game state with environment
        self.sync_game_state()
        
        # Log action results
        self.update_log(f"Action: {action}")
        self.update_log(f"Reward: {reward:.2f}")
        
        if info.get('is_noble_visit'):
            # Compare with previous state to find new noble
            current_noble_cards = set(self.env.game.player1.noble_cards)
            new_nobles = current_noble_cards - prev_noble_cards
            if new_nobles:
                new_noble = list(new_nobles)[0]
                noble_info = f"Noble {new_noble.name} visited! (+{new_noble.victory_point} VP)"
                self.update_log(noble_info)
        
        # Only reset if the game is actually finished (victory achieved)
        if done and self.env.game.player1.sum_victory_point >= self.env.target_vp:
            self.update_log("Game finished! Victory achieved!")
            
            # Reset environment
            self.obs, _ = self.env.reset()
            
            # Reset game state completely
            self.game.reset()
            
            # Sync the states after reset
            self.sync_game_state()
            
            # Reset reward
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
                
                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.button:
                        self.take_ai_action()
                        game_state_changed = True
                
                self.gui_manager.process_events(event)
            
            # Only redraw game state if it changed
            if game_state_changed:
                self.background.fill((200, 200, 200))
                self.gui.draw(self.background)
                game_state_changed = False
            
            # Draw everything in order
            self.gui.screen.blit(self.background, (0, 0))
            self.gui_manager.draw_ui(self.gui.screen)
            
            # Single flip per frame
            pygame.display.flip()
            
            # Update UI
            self.gui_manager.update(time_delta)

def main():
    app = AIControlledGUI()
    app.run()

if __name__ == "__main__":
    main() 