class RandomStartTwoPlayerEnv(gym.Env):
    """Two player environment with random starting positions"""
    def __init__(self, opponent_policy, verbose_dict=None):
        """Initialize environment with opponent policy
        
        Args:
            opponent_policy: Policy function that takes observation and returns action
            verbose_dict: Dictionary of verbosity settings
        """
        if verbose_dict is None:
            verbose_dict = get_verbose_dict()
        
        self.verbose = verbose_dict['env']
        if self.verbose:
            self.logger = TestLogger('logs/env')
            
        # Store opponent policy (required)
        if opponent_policy is None:
            raise ValueError("opponent_policy cannot be None")
        self.opponent_policy = opponent_policy
            
        # Initialize game
        self.game = Game(verbose_dict)  # This will create player1
        
        # Add second player if not already present
        if len(self.game.players) == 1:
            self.game.add_player("player2")
        
        # Verify both players are initialized
        assert len(self.game.players) == 2, "Game must have exactly 2 players"
                
        # Action and observation spaces
        action_n = self.game.players[0].num_actions
        self.action_space = spaces.Discrete(action_n)
        
        obs_space = get_observation_space(self.game)
        self.observation_space = spaces.Box(
            low=0,
            high=HIGH_VALUE,
            shape=(obs_space.shape[0],),
            dtype=np.int32
        )

        # Parameters
        self.target_vp = 15
        self.max_step = 127

    def step(self, action):
        # Execute learning player's action
        player_idx = 0 if self.player_starts_first else 1
        current_player = self.game.players[player_idx]
        
        # Convert action to scalar if it's a tensor
        if torch.is_tensor(action):
            action = action.item()
        
        victory_point, over_coin_count, got_card, noble_visit = current_player.do_action(self.game.board, action)
        
        # Check if game ended after player's move
        terminated, reward, info = self._check_game_end(player_idx, victory_point)
        
        # Let opponent make a move if game not over
        if not terminated:
            # Get opponent's observation and action
            opponent_idx = 1 - player_idx
            opponent_obs = self.get_observation(opponent_idx)
            opponent_action = self.opponent_policy(opponent_obs)
            
            # Execute opponent's action
            opp_vp, _, _, _ = self.game.players[opponent_idx].do_action(self.game.board, opponent_action)
            
            # Check if game ended after opponent's move
            terminated, reward, info = self._check_game_end(player_idx, victory_point, opp_vp)
        
        # Get observation for next state
        observation = self.get_observation(player_idx)
        
        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        """Reset environment and randomly determine player order"""
        if seed is not None:
            np.random.seed(seed)
        
        # Randomly decide if trained agent starts first
        self.player_starts_first = bool(np.random.randint(2))
        
        self.game.reset()
        
        # Make sure second player is added after reset
        if len(self.game.players) == 1:
            self.game.add_player("player2")
        
        # Let opponent make first move if player goes second
        player_idx = 0 if self.player_starts_first else 1
        if not self.player_starts_first:
            opponent_obs = self.get_observation(0)  # Get observation for opponent
            opponent_action = self.opponent_policy(opponent_obs)
            self.game.players[0].do_action(self.game.board, opponent_action)
        
        # Get observation from correct perspective
        observation = self.get_observation(player_idx)
        info = {"starts_first": self.player_starts_first}
        
        return observation, info 

    def get_observation(self, player_idx=0, verbose=False):
        """Get observation from specified player's perspective"""
        # Get observation containing both players' states
        obs = get_observation(self.game, player_idx, verbose=verbose, logger=self.logger if self.verbose else None)
        
        # Get action mask for current player
        action_mask = self.get_action_mask(player_idx)
        
        # Combine observation and action mask
        final_obs = np.concatenate([obs, action_mask])
        
        if verbose and self.logger:
            self.logger.info("Action Mask:")
            self.logger.info(f"Valid actions: {np.where(action_mask)[0]}")
            self.logger.info(f"Final shape: {final_obs.shape}")
            self.logger.info("-" * 50)
        
        return final_obs 