{-# LANGUAGE TopLevelKindSignatures #-}

module TLKS_Fail012 where

import Data.Kind (Type, Constraint)

type C :: Type -> Type -> Constraint
class C a where
