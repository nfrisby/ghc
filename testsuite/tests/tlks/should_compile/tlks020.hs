{-# LANGUAGE TopLevelKindSignatures #-}
{-# LANGUAGE PolyKinds, ExplicitForAll #-}

module TLKS_020 where

import Data.Kind (Type)

type T :: forall k. k -> forall j. j -> Type
data T (x :: hk) :: hj -> Type
