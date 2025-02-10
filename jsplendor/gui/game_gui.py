import pygame
import os

class SplendorGUI:
    def __init__(self, game):
        pygame.init()
        self.game = game
        
        # Window settings
        self.WINDOW_WIDTH = 1600
        self.WINDOW_HEIGHT = 1000
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
        self.CARD_WIDTH = 120
        self.CARD_HEIGHT = 160
        self.CARD_MARGIN = 20

        # Add card background colors for each level - alternative more saturated version
        self.CARD_COLORS = {
            1: (255, 200, 200),  # Stronger pink for level 1
            2: (200, 255, 200),  # Stronger mint green for level 2
            3: (200, 200, 255),  # Stronger light blue for level 3
        }

        # Add log box settings
        self.LOG_WIDTH = 400
        self.LOG_HEIGHT = 200
        self.LOG_X = self.WINDOW_WIDTH - self.LOG_WIDTH - 20  # 20px margin from right
        self.LOG_Y = self.WINDOW_HEIGHT - self.LOG_HEIGHT - 20  # 20px margin from bottom
        self.log_messages = ["Game Started"]  # List to store log messages
        self.MAX_LOG_LINES = 10  # Maximum number of lines to show

        # Load assets
        self.load_assets()

    def load_assets(self):
        # Create asset directory if it doesn't exist
        if not os.path.exists('jsplendor/gui/assets'):
            os.makedirs('jsplendor/gui/assets')

        # Initialize font
        self.font = pygame.font.Font(None, 24)

    def draw_gem(self, color, x, y, radius):
        # Draw gem border
        pygame.draw.circle(self.screen, self.GEM_BORDERS[color], (x, y), radius)
        # Draw gem interior
        pygame.draw.circle(self.screen, self.COLORS[color], (x, y), radius - 2)

    def draw_card(self, card, x, y):
        if card is None:
            # Draw card back instead of gray rectangle
            pygame.draw.rect(self.screen, (139, 69, 19), (x, y, self.CARD_WIDTH, self.CARD_HEIGHT))  # Brown color
            pygame.draw.rect(self.screen, (101, 67, 33), (x, y, self.CARD_WIDTH, self.CARD_HEIGHT), 2)  # Darker brown border
            # Add some decoration to card back
            pygame.draw.rect(self.screen, (101, 67, 33), 
                           (x + 10, y + 10, self.CARD_WIDTH - 20, self.CARD_HEIGHT - 20), 2)
            return

        # Draw card background with level-specific color
        bg_color = self.CARD_COLORS[card.level]
        pygame.draw.rect(self.screen, bg_color, (x, y, self.CARD_WIDTH, self.CARD_HEIGHT))
        pygame.draw.rect(self.screen, self.BLACK, (x, y, self.CARD_WIDTH, self.CARD_HEIGHT), 2)

        # Draw card level
        level_text = self.font.render(f"Level {card.level}", True, self.BLACK)
        self.screen.blit(level_text, (x + 5, y + 5))

        # Draw victory points
        vp_text = self.font.render(f"VP: {card.victory_point}", True, self.BLACK)
        self.screen.blit(vp_text, (x + 5, y + 25))

        # Draw gem color with border
        self.draw_gem(card.gem_color, 
                     x + self.CARD_WIDTH - 20, 
                     y + 20, 
                     12)

        # Draw price
        y_offset = 50
        for color, amount in zip(['WHITE', 'BLUE', 'GREEN', 'RED', 'BLACK'], card.price):
            if amount > 0:
                # Draw small gem icon
                self.draw_gem(color, x + 15, y + y_offset + 8, 8)
                # Draw amount with white text for dark backgrounds
                text_color = self.BLACK
                price_text = self.font.render(str(amount), True, text_color)
                self.screen.blit(price_text, (x + 30, y + y_offset))
                y_offset += 20

    def draw_coins(self, coins, x, y):
        for i, (color, count) in enumerate(coins.items()):
            # Draw coin with border
            self.draw_gem(color, x + i * 60, y, 25)
            # Draw count with white text for dark backgrounds
            text_color = self.WHITE if color in ['BLACK', 'BLUE'] else self.BLACK
            count_text = self.font.render(str(count), True, text_color)
            text_rect = count_text.get_rect(center=(x + i * 60, y))
            self.screen.blit(count_text, text_rect)

    def draw_noble(self, noble, x, y):
        if noble:
            # Draw noble card background
            pygame.draw.rect(self.screen, (255, 223, 196), (x, y, 120, 120))
            
            # Draw victory points
            noble_text = self.font.render(f"Noble: {noble.victory_point}VP", True, self.BLACK)
            self.screen.blit(noble_text, (x + 10, y + 10))
            
            # Draw required gems
            y_offset = 40
            for color, amount in zip(['WHITE', 'BLUE', 'GREEN', 'RED', 'BLACK'], noble.price):
                if amount > 0:
                    # Draw small gem icon
                    self.draw_gem(color, x + 25, y + y_offset, 8)
                    # Draw amount
                    price_text = self.font.render(str(amount), True, self.BLACK)
                    self.screen.blit(price_text, (x + 45, y + y_offset - 5))
                    y_offset += 20

    def add_log_message(self, message):
        self.log_messages.append(message)
        if len(self.log_messages) > self.MAX_LOG_LINES:
            self.log_messages.pop(0)

    def draw_log_box(self):
        # Draw log box background
        pygame.draw.rect(self.screen, self.WHITE, 
                        (self.LOG_X, self.LOG_Y, self.LOG_WIDTH, self.LOG_HEIGHT))
        pygame.draw.rect(self.screen, self.BLACK, 
                        (self.LOG_X, self.LOG_Y, self.LOG_WIDTH, self.LOG_HEIGHT), 2)

        # Draw "Game Log" header
        log_header = self.font.render("Game Log", True, self.BLACK)
        self.screen.blit(log_header, (self.LOG_X + 10, self.LOG_Y + 5))

        # Draw log messages
        for i, message in enumerate(self.log_messages):
            text = self.font.render(message, True, self.BLACK)
            self.screen.blit(text, (self.LOG_X + 10, self.LOG_Y + 30 + i * 20))

    def draw(self):
        # Fill background
        self.screen.fill((200, 200, 200))

        # Draw step counter at top center
        step_text = self.font.render(f"Step: {self.game.step}", True, self.BLACK)
        text_rect = step_text.get_rect(center=(self.WINDOW_WIDTH // 2, 30))
        self.screen.blit(step_text, text_rect)

        # Fixed y-positions for different sections
        NOBLE_Y = 80
        LEVEL3_Y = 250
        LEVEL2_Y = 460
        LEVEL1_Y = 670
        PLAYER_CARDS_Y = 880
        
        # Board coins position on right side
        BOARD_COINS_Y = 300
        PLAYER_COINS_Y = 500

        # Draw noble cards
        for i, noble in enumerate(self.game.board.noble_cards):
            self.draw_noble(noble, 100 + i * 160, NOBLE_Y)

        # Draw development cards
        level_positions = [
            (self.game.board.table_level3, "Level 3", LEVEL3_Y),
            (self.game.board.table_level2, "Level 2", LEVEL2_Y),
            (self.game.board.table_level1, "Level 1", LEVEL1_Y),
        ]
        
        for level_cards, label, y_pos in level_positions:
            # Draw level label
            label_text = self.font.render(label, True, self.BLACK)
            self.screen.blit(label_text, (100, y_pos - 25))
            
            # Draw deck first (card back)
            self.draw_card(None, 100, y_pos)
            
            # Draw available cards
            for i, card in enumerate(level_cards):
                self.draw_card(card, 100 + (i + 1) * (self.CARD_WIDTH + self.CARD_MARGIN), y_pos)

        # Draw board coins
        board_coins_text = self.font.render("Board Coins:", True, self.BLACK)
        self.screen.blit(board_coins_text, (1000, BOARD_COINS_Y - 50))
        self.draw_coins(self.game.board.coins, 1000, BOARD_COINS_Y)

        # Draw player coins
        player_coins_text = self.font.render("Player Coins:", True, self.BLACK)
        self.screen.blit(player_coins_text, (1000, PLAYER_COINS_Y - 50))
        self.draw_coins(self.game.player1.coins, 1000, PLAYER_COINS_Y)

        # Draw player cards at the bottom
        player_cards_text = self.font.render("Player Cards:", True, self.BLACK)
        self.screen.blit(player_cards_text, (100, PLAYER_CARDS_Y - 25))
        for i, card in enumerate(self.game.player1.development_cards[:5]):
            self.draw_card(card, 100 + i * (self.CARD_WIDTH + self.CARD_MARGIN), PLAYER_CARDS_Y)

        # Draw log box at the end
        self.draw_log_box()

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Handle mouse clicks here
                    pass

            self.draw()

        pygame.quit() 