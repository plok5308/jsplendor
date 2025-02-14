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

        # Draw card level
        level_text = self.font.render(f"Level {card.level}", True, self.BLACK)
        target_surface.blit(level_text, (x + 5, y + 5))

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

    def handle_scroll(self, event, mouse_pos):
        if not hasattr(self, 'log_messages'):
            return

        log_rect = pygame.Rect(
            self.WINDOW_WIDTH - 350,
            self.WINDOW_HEIGHT - 400,
            300,
            280
        )
        scroll_rect = self.get_scroll_bar_rect()

        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check if click is on scrollbar
            if scroll_rect.collidepoint(mouse_pos):
                self.scroll_dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.scroll_dragging = False
        elif event.type == pygame.MOUSEMOTION and self.scroll_dragging:
            # Update scroll position based on mouse movement
            _, mouse_y = mouse_pos
            scroll_area_height = log_rect.height - 40  # Subtract header height
            max_scroll = max(0, len(self.log_messages) - self.visible_lines)
            
            # Calculate relative position
            relative_y = (mouse_y - (log_rect.y + 40)) / (scroll_area_height - self.scroll_bar_width)
            self.scroll_y = int(max_scroll * relative_y)
            self.scroll_y = max(0, min(self.scroll_y, max_scroll))

    def get_scroll_bar_rect(self):
        if not hasattr(self, 'log_messages'):
            return None

        log_rect = pygame.Rect(
            self.WINDOW_WIDTH - 350,
            self.WINDOW_HEIGHT - 400,
            300,
            280
        )
        
        # Calculate scrollbar dimensions
        scroll_area_height = log_rect.height - 40  # Subtract header height
        total_lines = len(self.log_messages)
        
        if total_lines <= self.visible_lines:
            scroll_height = scroll_area_height
        else:
            scroll_height = (self.visible_lines / total_lines) * scroll_area_height
            scroll_height = max(20, scroll_height)  # Minimum scrollbar height
        
        # Calculate scrollbar position
        max_scroll = max(0, total_lines - self.visible_lines)
        if max_scroll == 0:
            scroll_pos = 0
        else:
            scroll_pos = (self.scroll_y / max_scroll) * (scroll_area_height - scroll_height)
        
        return pygame.Rect(
            log_rect.right - self.scroll_bar_width - 5,  # 5px padding from right
            log_rect.y + 40 + scroll_pos,  # Start below header
            self.scroll_bar_width,
            scroll_height
        )

    def draw(self, surface=None):
        target_surface = surface if surface is not None else self.screen
        
        # Fill background
        target_surface.fill((200, 200, 200))

        # Draw step counter and victory points at top center
        step_text = self.font.render(f"Step: {self.game.step}", True, self.BLACK)
        vp_text = self.font.render(f"Victory Points: {self.game.player1.sum_victory_point}", True, self.BLACK)
        
        # Position step counter slightly to the left of center
        step_rect = step_text.get_rect(center=(self.WINDOW_WIDTH // 2 - 100, 30))
        # Position victory points slightly to the right of center
        vp_rect = vp_text.get_rect(center=(self.WINDOW_WIDTH // 2 + 100, 30))
        
        target_surface.blit(step_text, step_rect)
        target_surface.blit(vp_text, vp_rect)

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

        # Draw AI button if it exists
        if hasattr(self, 'ai_button_rect'):
            pygame.draw.rect(target_surface, (100, 100, 255), self.ai_button_rect)
            text_surface = self.font.render("AI Action", True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=self.ai_button_rect.center)
            target_surface.blit(text_surface, text_rect)

        # Draw log if it exists
        if hasattr(self, 'log_messages'):
            log_rect = pygame.Rect(
                self.WINDOW_WIDTH - 350,  # Position on right side
                self.WINDOW_HEIGHT - 400,  # Position above button
                300,  # Width
                280   # Height
            )
            # Draw log background
            pygame.draw.rect(target_surface, (200, 200, 200), log_rect)
            pygame.draw.rect(target_surface, (100, 100, 100), log_rect, 2)  # Add border
            
            # Draw "Game Log" header
            header_surface = self.font.render("Game Log", True, (0, 0, 0))
            target_surface.blit(header_surface, (log_rect.x + 10, log_rect.y + 10))
            
            # Create a clip area for the messages
            message_area = pygame.Rect(
                log_rect.x + 10,
                log_rect.y + 40,
                log_rect.width - self.scroll_bar_width - 15,  # Leave space for scrollbar
                log_rect.height - 50
            )
            
            # Draw visible messages
            visible_messages = self.log_messages[self.scroll_y:self.scroll_y + self.visible_lines]
            for i, message in enumerate(visible_messages):
                text_surface = self.font.render(message, True, (0, 0, 0))
                y_pos = message_area.y + i * self.line_height
                if message_area.y <= y_pos <= message_area.bottom:
                    target_surface.blit(text_surface, (message_area.x, y_pos))
            
            # Draw scrollbar
            scroll_rect = self.get_scroll_bar_rect()
            if scroll_rect:
                pygame.draw.rect(target_surface, (150, 150, 150), scroll_rect)
                pygame.draw.rect(target_surface, (100, 100, 100), scroll_rect, 1)

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Handle mouse clicks here
                    self.handle_scroll(event, event.pos)

            self.draw()

        pygame.quit() 